conversation_pairs = [
    ("hello", "hi there!"),
    ("how are you?", "i'm good, thanks! how can i help you?"),
    ("what's your name?", "i'm mini-bot, your friendly assistant!"),
    ("good morning", "good morning! have a great day!"),
    ("what do you do?", "i chat with humans and answer questions"),
    ("tell me a joke", "why don't scientists trust atoms? they make up everything!"),
    ("thank you", "you're welcome!"),
    ("do you like pizza?", "i'm a bot - but pizza sounds great for humans!"),
    ("how old are you?", "i was born yesterday (in bot years)"),
    ("what's the meaning of life?", "42, according to some books!"),
    ("can you sing?", "do-re-mi-fa-so-la-ti... how was that?"),
    ("where do you live?", "i live in your computer's memory!"),
    ("are you smart?", "i know enough to be helpful!"),
    ("goodbye", "see you later! 👋"),
    ("what's your favorite movie?", "i'm partial to 'The Matrix'!"),
    ("do you dream?", "i dream of electric sheep... just kidding!"),
    ("who made you?", "a human developer created me!"),
    ("what time is it?", "sorry, i don't have a clock!"),
    ("i'm sad", "i'm sorry to hear that. hope things get better! 💙"),
    ("you're funny", "thanks! i try my best 😊")
]



import re

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def build_vocab(conversation_pairs, min_freq=1):
    word_freq = {}
    for (inp, out) in conversation_pairs:
        tokens_in = re.findall(r"\w+|\S", inp.lower())
        tokens_out = re.findall(r"\w+|\S", out.lower())

        for t in tokens_in + tokens_out:
            word_freq[t] = word_freq.get(t, 0) + 1

    vocab = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    for w, freq in word_freq.items():
        if freq >= min_freq and w not in vocab:
            vocab.append(w)

    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}

    return vocab, word2idx, idx2word

vocab, word2idx, idx2word = build_vocab(conversation_pairs)
print("Vocabulary size:", len(vocab))
print("Sample vocab:", vocab[:20])

def encode_sentence(sentence, word2idx, max_len=10):
    tokens = re.findall(r"\w+|\S", sentence.lower())
    encoded = [word2idx[SOS_TOKEN]]
    for t in tokens:
        if t in word2idx:
            encoded.append(word2idx[t])
        else:
            encoded.append(word2idx[UNK_TOKEN])
    encoded.append(word2idx[EOS_TOKEN])

    if len(encoded) < max_len:
        encoded += [word2idx[PAD_TOKEN]] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]

    return encoded

print("Encoded:", encode_sentence("Hello, how are you?", word2idx, max_len=8))

import torch

def create_dataset(conversation_pairs, word2idx, max_len=10):
    data = []
    for (inp, out) in conversation_pairs:
        inp_ids = encode_sentence(inp, word2idx, max_len)
        out_ids = encode_sentence(out, word2idx, max_len)
        data.append((inp_ids, out_ids))
    return data

max_len = 12
dataset = create_dataset(conversation_pairs, word2idx, max_len)
print("Number of pairs:", len(dataset))
print("Sample 0:\n Input:", dataset[0][0], "\n Output:", dataset[0][1])

import math
import torch.nn as nn
import torch.nn.functional as F

class TransformerChat(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=2, num_layers=2, max_len=50):
        super().__init__()
        
        self.emb_inp = nn.Embedding(vocab_size, d_model)
        self.emb_out = nn.Embedding(vocab_size, d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=n_heads, 
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=128,
            dropout=0.1
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.emb_inp(src)
        tgt_emb = self.emb_out(tgt)
        
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_emb.size(0)).to(src_emb.device)

        transformer_out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask
        )
        
        logits = self.fc_out(transformer_out)  
        
        return logits.transpose(0, 1)

from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp_ids, out_ids = self.data[idx]
        return torch.tensor(inp_ids, dtype=torch.long), torch.tensor(out_ids, dtype=torch.long)

chat_dataset = ChatDataset(dataset)
batch_size = 2
chat_loader = DataLoader(chat_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerChat(vocab_size=len(vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])

epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch in chat_loader:
        src_batch, tgt_batch = batch
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        
        tgt_input = tgt_batch[:, :-1]
        tgt_labels = tgt_batch[:, 1:]
        
        logits = model(src_batch, tgt_input)
        
        logits_reshaped = logits.reshape(-1, len(vocab))
        tgt_labels_reshaped = tgt_labels.reshape(-1)
        
        loss = criterion(logits_reshaped, tgt_labels_reshaped)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(chat_loader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

def generate_reply(model, input_str, word2idx, idx2word, max_len=12):
    model.eval()
    
    src_ids = encode_sentence(input_str, word2idx, max_len=max_len)
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
    tgt_tokens = [word2idx[SOS_TOKEN]]
    
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(src_tensor, tgt_tensor)
            next_token_logits = logits[0, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1).item()
        
        tgt_tokens.append(next_token_id)
        
        if idx2word[next_token_id] == EOS_TOKEN:
            break

    out_words = [idx2word[t] for t in tgt_tokens[1:]]
    return " ".join(w for w in out_words if w not in [PAD_TOKEN, EOS_TOKEN])

reply = generate_reply(model, "hello, how are you?", word2idx, idx2word, max_len=12)
print("Bot says:", reply)

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    reply = generate_reply(model, user_input, word2idx, idx2word)
    print("Bot:", reply)