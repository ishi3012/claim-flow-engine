from datetime import datetime
from .denial_schema import Denial
import pandas as pd

# Track claims to detect duplicate claims
claims_seen = set()

def rule_duplicate_claim(claim: pd.Series) -> pd.Series:
    """
    Denies if the same patient has already had a procedure 
    with the same CPT code on the same date. 
    
    Parameters:
        claim (pd.Series): A single claim claim
    """
    claims_seen = set()
    key = (claim['patient_id'], claim['procedure_date'], claim['cpt_code'])

    if key in claims_seen:
        return Denial(reason="Duplicate Claim", code="18", severity=2, auto_appealable=False, routing_tag="Validation")    
    claims_seen.add(key)
    return None

def rule_authorization_required(claim: pd.Series) -> pd.Series:
    """
    Denies claims with high-cost procedures (billed > $1000) or CPT codes starting 
    with '7' (e.g., radiology, surgery) that typically require prior authorization.

    Parameters:
        claim (pd.Series): A single claim claim
    """

    if claim['cpt_code'].startswith('7') or claim['billed_amount'] > 1000:
        return Denial(reason="Authorization Required", code="197", severity=4, auto_appealable=True, routing_tag="PreAuth")
    return None

def rule_medical_necessity(claim: pd.Series) -> pd.Series:
    """
    Flags claims with vague diagnoses (ICD-10 'R' codes) or diagnostic CPT codes 
    used for children, which often fail medical necessity checks.

    Parameters:
        claim (pd.Series): A single claim claim
    """
    if claim['icd10_code'].startswith('R') or (80000 <= int(claim['cpt_code']) <= 89999 and claim['patient_age'] < 18):
        return Denial(reason="Medical Necessity", code="50", severity=3, auto_appealable=True, routing_tag="Clinical")
    return None

def rule_timely_filing(claim: pd.Series) -> pd.Series:
    """
    Denies claims submitted more than 90 days after the procedure date, 
    simulating timely filing limits from payers.

    Parameters:
        claim (pd.Series): A single claim claim
    """
    if "days_since_procedure" in claim and claim["days_since_procedure"] > 90:
        return Denial(reason="Timely Filing", code="29", severity=4, auto_appealable=False, routing_tag="Compliance")
    return None

def rule_code_mismatch(claim: pd.Series) -> pd.Series:
    """
      Incompatible CPT and ICD-10 combinations 
        
        Parameters:
        claim (pd.Series): A single claim claim
    """
    if claim["cpt_code"].startswith("3") and claim["icd10_code"].startswith("F"):
        return Denial(reason="Code Mismatch", code="11", severity=3, auto_appealable=False, routing_tag="Coding")
    return None


# Rules Registry
denial_rules = [
    rule_duplicate_claim,
    rule_authorization_required,
    rule_medical_necessity,
    rule_code_mismatch,
    rule_timely_filing
]

# apply denial rules to claim

def apply_denial_rules(claim: pd.Series) -> pd.Series:
    """
    Applies all denial rules to the claim and returns the most severe matching denial, if any.

    Parameters:
        claim (pd.Series): A single claim record

    Returns:
        pd.Series: Updated with denial fields if applicable
    """
    matching_denials = []
    for rule in denial_rules:
        result = rule(claim)
        if result:
            matching_denials.append(result)

    if matching_denials:
        selected = max(matching_denials, key=lambda d: d.severity)
        claim['denial_flag'] = 1
        claim['denial_reason'] = selected.reason
        claim['denial_code'] = selected.code
        claim['denial_severity'] = selected.severity
        claim['auto_appealable'] = selected.auto_appealable
        claim['routing_tag'] = selected.routing_tag
        return claim
    else:
        claim["denial_flag"] = 0
        claim["denial_reason"] = None
        claim["denial_code"] = None
        claim["denial_severity"] = 0
        claim["auto_appealable"] = False
        claim["routing_tag"] = None
    return claim





