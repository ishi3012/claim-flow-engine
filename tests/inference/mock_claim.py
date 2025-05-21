import pandas as pd

# Minimal viable input for feature engineering
mock_input = pd.DataFrame(
    [
        {
            "claim_id": 1,
            "submission_date": "2024-12-08",
            "service_date": "2024-12-01",
            "denial_date": "2024-12-15",
            "denied": 1,
            "denial_reason": "Duplicate claim",
            "payer_id": "PAYER004",
            "provider_type": "Independent",
            "resubmission": 0,
            "followup_notes": "Requested auth code",
            "total_charge_amount": 945.88,
            "patient_dob": "1993-09-22",
            "prior_authorization": "Y",
            "accident_indicator": "N",
            "claim_type": "Professional",
            "billing_provider_specialty": "Oncology",
            "facility_code": 11,
            "plan_type": "private",
            "diagnosis_code": "Z00.00",
            "procedure_code": 85025,
        }
    ]
)
