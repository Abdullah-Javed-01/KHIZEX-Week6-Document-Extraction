"""Khizex Week 6: Automated Document Extraction (OCR + NLP)\nLegal-domain pipeline using only synthetic contract scans."""
from pathlib import Path
import re, time, json, difflib
import numpy as np
import pandas as pd
import cv2
import pytesseract
from pytesseract import Output
from rapidfuzz.distance import Levenshtein

SECTION_NAMES = ["PARTIES","EFFECTIVE DATE","PAYMENT","OBLIGATIONS","CONFIDENTIALITY","GOVERNING LAW","SIGNATURES"]

def deskew(gray):
    inv = cv2.bitwise_not(gray)
    coords = np.column_stack(np.where(inv > 120))
    if len(coords) < 100: return gray
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > 5: return gray
    h,w=gray.shape
    M=cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    return cv2.warpAffine(gray,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

def preprocess_image(path):
    gray=cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    gray=deskew(gray)
    gray=cv2.fastNlMeansDenoising(gray,None,7,7,21)
    # Estimate uneven page illumination, divide it out, then binarize.
    background=cv2.GaussianBlur(gray,(0,0),sigmaX=25,sigmaY=25)
    normalized=cv2.divide(gray,background,scale=255)
    _, bw=cv2.threshold(normalized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return bw

def ocr_image(image_or_path):
    if isinstance(image_or_path,(str,Path)):
        img=cv2.imread(str(image_or_path),cv2.IMREAD_GRAYSCALE)
    else: img=image_or_path
    data=pytesseract.image_to_data(img,output_type=Output.DATAFRAME,config='--psm 6')
    good=data[(data.conf>=0)&data.text.notna()&data.text.str.strip().astype(bool)]
    confidence=float(good.conf.mean()) if len(good) else 0.0
    text=pytesseract.image_to_string(img,config='--psm 6')
    return text.strip(), confidence

def normalize_text(s):
    s=s.lower().replace('“','"').replace('”','"')
    s=re.sub(r'\s+',' ',s)
    return s.strip()

def cer(ref,hyp):
    r=normalize_text(ref); h=normalize_text(hyp)
    return Levenshtein.distance(r,h)/max(1,len(r))

def wer(ref,hyp):
    r=normalize_text(ref).split(); h=normalize_text(hyp).split()
    return Levenshtein.distance(r,h)/max(1,len(r))

def parse_sections(text):
    # OCR may add punctuation around headings, so find heading lines fuzzily by normalized line.
    sections={}; current='HEADER'; sections[current]=[]
    for line in text.splitlines():
        clean=re.sub(r'[^A-Z ]','',line.upper()).strip()
        matched=None
        for s in SECTION_NAMES:
            if clean==s or (len(clean)>4 and difflib.SequenceMatcher(None,clean,s).ratio()>=0.82):
                matched=s; break
        if matched:
            current=matched; sections.setdefault(current,[])
        else: sections.setdefault(current,[]).append(line.strip())
    return {k:' '.join(v).strip() for k,v in sections.items()}

def extract_fields(text):
    """Context-aware field extraction. It uses section context plus patterns, not global keyword matching."""
    sections=parse_sections(text)
    alltxt=' '.join(sections.values())
    out={k:'' for k in ['party_a','party_b','effective_date','amount','governing_law','obligation']}
    p=sections.get('PARTIES','')
    m=re.search(r'between\s+(.+?)\s+\(["“]?Provider["”]?\)\s+and\s+(.+?)\s+\(["“]?Client["”]?\)',p,re.I)
    if m: out['party_a'],out['party_b']=m.group(1).strip(' .'),m.group(2).strip(' .')
    e=sections.get('EFFECTIVE DATE','')
    m=re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',e,re.I)
    if m: out['effective_date']=m.group(0)
    pay=sections.get('PAYMENT','')
    m=re.search(r'\$\s?[\d,]+(?:\.\d{2})?',pay)
    if m: out['amount']=m.group(0).replace(' ','')
    gov=sections.get('GOVERNING LAW','')
    m=re.search(r'(?:State of|laws of)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)',gov)
    if m: out['governing_law']=m.group(1)
    ob=sections.get('OBLIGATIONS','')
    m=re.search(r'Provider shall\s+(.+?)(?:\.\s+The Client|\.|$)',ob,re.I)
    if m: out['obligation']=m.group(1).strip()
    return out, sections

def sentence_split(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', re.sub(r'\s+',' ',text)) if len(s.strip())>20]

def extractive_summary(text, max_sent=4):
    sents=sentence_split(text)
    if not sents: return text[:500]
    words=re.findall(r'[A-Za-z]{3,}',text.lower())
    stop={'the','and','this','that','shall','with','from','into','under','each','both','only','will','made','service','agreement'}
    freq={}
    for w in words:
        if w not in stop: freq[w]=freq.get(w,0)+1
    scored=[]
    for i,s in enumerate(sents):
        toks=re.findall(r'[A-Za-z]{3,}',s.lower())
        score=sum(freq.get(w,0) for w in toks)/max(1,len(toks))
        scored.append((score,i,s))
    chosen=sorted(sorted(scored,reverse=True)[:max_sent],key=lambda x:x[1])
    return ' '.join(x[2] for x in chosen)

def field_aware_summary(fields):
    bits=[]
    if fields['party_a'] and fields['party_b']:
        bits.append(f"The agreement is between {fields['party_a']} and {fields['party_b']}.")
    if fields['effective_date']: bits.append(f"It is effective on {fields['effective_date']}.")
    if fields['amount']: bits.append(f"The stated service fee is {fields['amount']}.")
    if fields['obligation']: bits.append(f"The provider is required to {fields['obligation']}.")
    if fields['governing_law']: bits.append(f"The governing law is {fields['governing_law']}.")
    return ' '.join(bits[:5])

def lcs_len(a,b):
    n=len(b); dp=[0]*(n+1)
    for x in a:
        prev=0
        for j,y in enumerate(b,1):
            tmp=dp[j]
            if x==y: dp[j]=prev+1
            else: dp[j]=max(dp[j],dp[j-1])
            prev=tmp
    return dp[-1]

def rouge_l(ref,hyp):
    a=normalize_text(ref).split(); b=normalize_text(hyp).split()
    if not a or not b:return 0.0
    l=lcs_len(a,b); p=l/len(b); r=l/len(a)
    return 0 if p+r==0 else 2*p*r/(p+r)
