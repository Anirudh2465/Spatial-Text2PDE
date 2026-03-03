import re
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import torch

class PLCSScorer:
    def __init__(self):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        # We use a fast, small model to compute BERTScore quickly
        self.bert_model_type = "distilbert-base-uncased"
        
    def extract_physics(self, text):
        # Radius
        m_rad = re.search(r"radius(?:\s*of|\s*:)?\s*([\d\.]+)", text)
        rad = float(m_rad.group(1).rstrip('.')) if m_rad else None
        
        # Position X, Y
        m_pos = re.search(r"(?:position:|located at \(|positioned at X=|coordinates |position \(|cylinder at |at\s*\(?|X=)\s*([\d\.]+)(?:,\s*Y=|, |,\s*)([\d\.]+)", text)
        px = float(m_pos.group(1).rstrip('.')) if m_pos else None
        py = float(m_pos.group(2).rstrip('.')) if m_pos else None
        
        # Velocity
        m_vel = re.search(r"velocity(?:\s*of|\s*is|\s*at the inlet is|\s*=)?\s*([\d\.]+)", text)
        v = float(m_vel.group(1).rstrip('.')) if m_vel else None
        
        # Reynolds
        m_re = re.search(r"(?:Reynolds number\s*is|Reynolds number\s*of|Re\s*=)\s*([\d\.]+)", text)
        re_num = float(m_re.group(1).rstrip('.')) if m_re else None
        
        # Flow
        m_flow = re.search(r"(laminar|transitioning(?:\s+in\s+the\s+wake)?|turbulent)", text)
        flow = m_flow.group(1).strip() if m_flow else None
        
        return {
            'radius': rad,
            'px': px,
            'py': py,
            'vel': v,
            're': re_num,
            'flow': flow
        }

    def compute_scores(self, y_true_text, y_pred_text):
        # --- 1. Component A - NLP Quality Score ---
        # BERTScore F1
        P, R, F1 = bert_score([y_pred_text], [y_true_text], lang="en", model_type=self.bert_model_type, verbose=False)
        bert_score_f1 = F1.item()
        
        # ROUGE-L
        rouge_scores = self.rouge.score(y_true_text, y_pred_text)
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        # Combined NLP Score
        nlp_score = (bert_score_f1 + rouge_l) / 2.0
        
        # --- 2. Component B - Physical Parameter Score ---
        true_phys = self.extract_physics(y_true_text)
        pred_phys = self.extract_physics(y_pred_text)
        
        # B1. Continuous Variables (Regression Score)
        continuous_keys = ['radius', 'px', 'py', 'vel', 're']
        reg_scores = []
        for k in continuous_keys:
            if true_phys[k] is not None and pred_phys[k] is not None:
                rel_err = abs(true_phys[k] - pred_phys[k]) / (abs(true_phys[k]) + 1e-8)
                score_i = max(0.0, 1.0 - rel_err)
            else:
                score_i = 0.0 # Missing prediction is penalized fully
            reg_scores.append(score_i)
            
        regression_score = sum(reg_scores) / len(continuous_keys)
        
        # B2. Flow Type (Classification Score)
        if true_phys['flow'] and pred_phys['flow'] and true_phys['flow'] == pred_phys['flow']:
            classification_score = 1.0
        else:
            classification_score = 0.0
            
        # B3. Physical Score
        physical_score = 0.8 * regression_score + 0.2 * classification_score
        
        # --- 3. Final Composite Score (PLCS) ---
        plcs = 0.7 * physical_score + 0.3 * nlp_score
        
        return {
            'plcs': plcs,
            'physical_score': physical_score,
            'regression_score': regression_score,
            'classification_score': classification_score,
            'nlp_score': nlp_score,
            'bert_score': bert_score_f1,
            'rouge_l': rouge_l,
            'extracted_true': true_phys,
            'extracted_pred': pred_phys
        }
