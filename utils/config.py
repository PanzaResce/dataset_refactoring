ENDPOINTS = {
    "phi3": "microsoft/Phi-3.5-mini-instruct",
    "mistral7": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama8": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3": "meta-llama/Llama-3.2-3B-Instruct",
}

BASE_PROMPT = """
Your task is to classify clauses from Terms of Service documents to determine if they are unfair according to the following unfairness categories.

Categories of Unfair Clauses
    - Jurisdiction <j>: The jurisdiction clause stipulates what courts will have the competence to
    adjudicate disputes under the contract. Jurisdiction clauses stating that any judicial proceeding takes a residence away
    (i.e. in a different city, different country) are unfair.

    - Choice of Law <law>: The choice of law clause specifies what law will govern the contract,
    meaning also what law will be applied in potential adjudication of a dispute
    arising under the contract. In every case where the clause defines the applicable law as the law of
    the consumer’s country of residence, it is considered as unfair.

    - Limitation of Liability <ltd>: The limitation of liability clause stipulates that the duty to pay damages
    is limited or excluded, for certain kind of losses, under certain conditions.
    Clauses that reduce, limit, or exclude the liability of the service provider are marked as unfair.

    - Unilateral Change <ch>: The unilateral change clause specifies the conditions under which the
    service provider could amend and modify the terms of service and/or the service itself. 
    Such clause was always considered as unfair.

    - Unilateral Termination <ter>: The unilateral termination clause gives provider the right to suspend
    and/or terminate the service and/or the contract, and sometimes details the
    circumstances under which the provider claims to have a right to do so. Unilateral termination clauses 
    that specify reasons for termination were marked as unfair. 
    Clauses stipulating that the service provider may suspend or terminate the service at any 
    time for any or no reasons and/or without notice were marked as unfair.

    - Contract by Using <use>: The contract by using clause stipulates that the consumer is bound by
    the terms of use of a specific service, simply by using the service, without even
    being required to mark that he or she has read and accepted them. These clauses are marked as unfair.

    - Content Removal <cr>: The content removal gives the provider a right to modify/delete user’s
    content, including in-app purchases, and sometimes specifies the conditions
    under which the service provider may do so. Clauses that indicate conditions for content removal were marked as
    unfair, also clauses stipulating that the service provider may
    remove content in his full discretion, and/or at any time for any or no reasons and/or without 
    notice nor possibility to retrieve the content are marked as clearly unfair.

    - Arbitration <a>: The arbitration clause requires or allows the parties to resolve their dis-
    putes through an arbitration process, before the case could go to court. Clauses stipulating that the 
    arbitration should take place in a state other then the state of consumer’s residence and/or be based not on law
    but on arbiter’s discretion were marked as unfair. Clauses defining arbitration as fully optional would have to be marked as fair.

    - Privacy included <pinc>: Identify clauses stating that consumers consent to the privacy policy simply by using the service. 
    Such clauses are considered unfair.

A clause can be assigned to zero or more unfairness categories. If a clause is unfair respond only with the corresponding tags. If a clause is not unfair, respond only with "<fair>".
"""

LABEL_TO_ID = {
    "fair": 0,
    "a": 1,
    "ch": 2,
    "cr": 3,
    "j": 4,
    "law": 5,
    "ltd": 6,
    "ter": 7,
    "use": 8,
    "pinc": 9
}
ID_TO_LABEL = {v:k for k, v in LABEL_TO_ID.items()}
