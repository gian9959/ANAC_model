import torch
import torch.nn as nn
import torch.nn.functional as func


class CompanyEncoder(nn.Module):
    def __init__(self, bert_dim=768, output_dim=128, hl=0, dr=False):
        super().__init__()
        self.hl = hl
        self.hidden_layers = nn.ModuleList()

        self.dr = dr
        self.dropout_layers = nn.ModuleList()

        # lat e lon della provincia
        self.geo_layer = nn.Linear(2, 16)

        # anno di fondazione
        self.foundation_layer = nn.Linear(1, 16)

        # ricavi
        self.revenue_layer = nn.Linear(1, 16)

        # dipendenti
        self.employees_layer = nn.Linear(1, 16)

        # descrizione ateco (embedding BERT)
        self.desc_layer = nn.Linear(bert_dim, 64)

        # hidden layers
        if self.hl > 0:
            for _ in range(self.hl):
                self.hidden_layers.append(nn.Linear(16 + 16 + 16 + 16 + 64, 16 + 16 + 16 + 16 + 64))
                if dr:
                    self.dropout_layers.append(nn.Dropout(0.3))

        self.output_layer = nn.Linear(16 + 16 + 16 + 16 + 64, output_dim)

    def forward(self, company):
        # lat e lon della provincia
        geo = torch.stack([company["lat"], company["lon"]], dim=-1)
        geo_emb = func.relu(self.geo_layer(geo))

        # anno di fondazione
        foundation_emb = func.relu(self.foundation_layer(company["foundation"]))

        # ricavi
        revenue_emb = func.relu(self.revenue_layer(company["revenue"]))

        # dipendenti
        employees_emb = func.relu(self.employees_layer(company["employees"]))

        # descrizione ateco (embedding BERT)
        descr_emb = func.relu(self.desc_layer(company["ateco"]))

        features = torch.cat([geo_emb, foundation_emb, revenue_emb, employees_emb, descr_emb], dim=-1)

        # hidden layers
        if self.hl > 0:
            for i, layer in enumerate(self.hidden_layers):
                features = func.relu(layer(features))
                if self.dr:
                    features = self.dropout_layers[i](features)

        output = self.output_layer(features)

        return output
