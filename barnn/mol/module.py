import torch
import lightning.pytorch as pl


class LanguageModel(pl.LightningModule):
    def __init__(self, model, vocabulary, tokenizer):
        super().__init__()
        self.model = model
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.metric = torch.nn.CrossEntropyLoss()

    def sample(self, batch_size, unroll=100, temperature=1):
        self.model.eval()
        start_token = torch.zeros(batch_size, dtype=torch.long, device=self.model.device)
        start_token[:] = self.vocabulary.vocabulary_map["[BEG]"]
        input_vector = start_token
        sequences = [
            self.vocabulary.vocabulary_map["[BEG]"] * torch.ones([batch_size, 1],
                                                    dtype=torch.long,
                                                    device=self.model.device)
        ]
        hidden_state = None
        finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.model.device)
        
        for _ in range(unroll - 1):  # First token already initialized as start token
            logits, hidden_state = self.model(input_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1) / temperature
            probabilities = logits.softmax(dim=1)
            new_input_vector = torch.multinomial(probabilities, 1).view(-1)
            new_input_vector = torch.where(finished_mask, torch.zeros_like(new_input_vector), new_input_vector)
            # Append the sampled token
            sequences.append(new_input_vector.view(-1, 1))
            # Update finished_mask when [END] token is generated
            end_token = self.vocabulary.vocabulary_map["[END]"]
            finished_mask = finished_mask | (new_input_vector == end_token)
            # Break early if all sequences are finished
            if finished_mask.all():
                break
            # Set the input for the next step
            input_vector = new_input_vector
        # Concatenate sequences and convert to tokens
        sequences = torch.cat(sequences, 1).tolist()
        tokens = self.vocabulary.labels2tokens(sequences)
        # Convert tokens to final output (e.g., sentences)
        self.model.train()
        return [self.tokenizer.untokenize(token) for token in tokens]

    def training_step(self, batch, batch_idx):
        x_prev, x_next = batch
        logits, _ = self.model(x_prev)
        loss = self.metric(
            logits.reshape(-1, logits.size(-1)), x_next.reshape(-1)
            )
        self.log("train_nll_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    def validation_step(self, batch, batch_idx):
        x_prev, x_next = batch
        logits, _ = self.model(x_prev)
        loss = self.metric(
            logits.reshape(-1, logits.size(-1)), x_next.reshape(-1)
            )
        self.log("val_nll_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    def predict_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0002, weight_decay=1e-8)

class BARNNLanguageModel(LanguageModel):
    def training_step(self, batch, batch_idx):
        nll = super().training_step(batch, batch_idx)
        kl = self.model.rnn.kl().mean()
        self.log("train_kll_loss", kl, prog_bar=True, on_epoch=True)
        return nll + kl
    def validation_step(self, batch, batch_idx):
        nll = super().validation_step(batch, batch_idx)
        kl = self.model.rnn.kl().mean()
        self.log("val_kll_loss", kl, prog_bar=True, on_epoch=True)
        return nll + kl