import torch
import torch.nn as nn
import torch.nn.functional as func


class CompanyEncoder(nn.Module):
    def __init__(self, bert_dim=768, output_dim=128):
        super().__init__()

        # lat e lon della provincia
        self.geo_layer = nn.Linear(2, 16)

        # anno di fondazione
        self.foundation_layer = nn.Linear(1, 16)

        # fatturato
        self.revenue_layer = nn.Linear(1, 16)

        # descrizione ateco (embedding BERT)
        self.desc_layer = nn.Linear(bert_dim, 64)

        self.output_layer = nn.Linear(16 + 16 + 16 + 64, output_dim)

    def forward(self, company):
        # lat e lon della provincia
        geo = torch.stack([company["lat"], company["lon"]], dim=1)
        geo_emb = func.relu(self.geo_layer(geo))

        # anno di fondazione
        foundation_emb = func.relu(self.foundation_layer(company["foundation"]))

        # fatturato
        revenue_emb = func.relu(self.revenue_layer(company["revenue"]))

        # descrizione ateco (embedding BERT)
        descr_emb = func.relu(self.desc_layer(company["ateco_desc"]))

        features = torch.cat([geo_emb, foundation_emb, revenue_emb, descr_emb], dim=1)
        output = self.output_layer(features)

        return output
