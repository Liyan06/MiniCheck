SYSTEM_PROMPT = """Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. Please assess the claim's consistency with the document by responding with either "Yes" or "No"."""


USER_PROMPT = """Document: [DOCUMENT]\nClaim: [CLAIM]"""