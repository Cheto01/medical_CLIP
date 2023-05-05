import os
import re

def parse_report(report_path):
    with open(report_path, 'r') as f:
        report = f.read()
        
    # Extract impression and remove from report
    impression_match = re.search(r'IMPRESSION:(.*)', report, re.DOTALL)
    impression = impression_match.group(1).strip() if impression_match else ""
    report = re.sub(r'IMPRESSION:(.*)', "", report, flags=re.DOTALL)
    
    # Aggregate technique and findings and rename as caption
    technique_match = re.search(r'TECHNIQUE:(.*)', report, re.DOTALL)
    technique = technique_match.group(1).strip() if technique_match else ""
    findings_match = re.search(r'FINDINGS:(.*)', report, re.DOTALL)
    findings = findings_match.group(1).strip() if findings_match else ""
    caption = technique + "\n" + findings
    
    # Remove titles
    report = re.sub(r'EXAMINATION:|INDICATION:|TECHNIQUE:|COMPARISON:|FINDINGS:|IMPRESSION:', "", report)
    
    # Remove unnecessary spaces and characters
    impression = re.sub(r'\s+', " ", impression)
    caption = re.sub(r'\s+', " ", caption)
    
    return caption, impression

def image_caption_impression_pairs(dataset_path):
    pairs = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".dcm.gz"):
                dicom_id = file.split(".")[0]
                study_id = os.path.basename(os.path.dirname(root))
                subject_id = os.path.basename(os.path.dirname(os.path.dirname(root)))
                report_path = os.path.join(root, study_id + ".txt")
                if os.path.exists(report_path):
                    caption, impression = parse_report(report_path)
                    pairs.append({"dicom_id": dicom_id, "study_id": study_id, "subject_id": subject_id, "caption": caption, "impression": impression})
    
    return pairs
