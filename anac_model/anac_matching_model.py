import torch.nn as nn

from anac_model.encoders.company_encoder import CompanyEncoder
from anac_model.encoders.tender_encoder import TenderEncoder


class AnacMatchingModel(nn.Module):
    def __init__(self, hidden_layers=0, dropout=False):
        super().__init__()
        self.tender_encoder = TenderEncoder(hl=hidden_layers, dr=dropout)
        self.company_encoder = CompanyEncoder(hl=hidden_layers, dr=dropout)

    def forward(self, tender, company):
        tender_emb = self.tender_encoder(tender)
        company_emb = self.company_encoder(company)

        scores = nn.functional.cosine_similarity(tender_emb, company_emb)
        return scores
