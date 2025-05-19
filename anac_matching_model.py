import torch.nn as nn

from encoders.company_encoder import CompanyEncoder
from encoders.tender_encoder import TenderEncoder


class AnacMatchingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tender_encoder = TenderEncoder()
        self.company_encoder = CompanyEncoder()

    def forward(self, tender, company):
        tender_emb = self.tender_encoder(tender)
        company_emb = self.company_encoder(company)

        scores = nn.functional.cosine_similarity(tender_emb, company_emb)
        return scores
