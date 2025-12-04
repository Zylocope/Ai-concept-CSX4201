conversation_pairs = [
    ("What is AI in business?", "Artificial Intelligence in business refers to the use of intelligent computer systems that can perform tasks typically requiring human intelligence, such as decision-making, pattern recognition, and natural language processing, to optimize business operations and drive growth."),
    
    ("What are the main applications of AI in business?", "Key AI applications in business include: 1) Customer service chatbots 2) Predictive analytics for sales 3) Automated marketing campaigns 4) Supply chain optimization 5) Fraud detection 6) HR recruitment automation 7) Business process automation 8) Market trend analysis"),
    
    ("How does AI improve customer service?", "AI improves customer service through 24/7 chatbot support, personalized recommendations, automated ticket routing, sentiment analysis of customer feedback, and predictive service interventions, resulting in faster response times and improved customer satisfaction."),
    
    ("What is predictive analytics in business?", "Predictive analytics uses AI and machine learning to analyze historical data and predict future business outcomes, such as sales trends, customer behavior, market changes, and potential risks, enabling data-driven decision making."),
    
    ("How can small businesses implement AI?", "Small businesses can implement AI through: 1) Cloud-based AI services 2) Pre-built AI solutions 3) Customer service chatbots 4) Marketing automation tools 5) Basic data analytics platforms. These options are cost-effective and require minimal technical expertise."),
    
    ("What are the costs of implementing AI?", "AI implementation costs vary widely, from affordable cloud-based solutions ($100-1000/month) to enterprise-level custom AI systems ($100,000+). Costs include software licenses, infrastructure, training, maintenance, and potential consulting fees."),
    
    ("What is machine learning in business?", "Machine learning in business is a subset of AI that enables systems to automatically learn and improve from experience. It's used for customer segmentation, demand forecasting, risk assessment, and process optimization without explicit programming."),
    
    ("How does AI affect employment?", "AI impacts employment by automating routine tasks, creating new job roles in AI management and development, and requiring workforce reskilling. While some jobs may be displaced, AI creates new opportunities in data science, AI ethics, and human-AI collaboration."),
    
    ("What are the risks of AI in business?", "Key risks of AI in business include: 1) Data privacy concerns 2) Algorithmic bias 3) Security vulnerabilities 4) Implementation costs 5) Integration challenges 6) Dependency on AI systems 7) Ethical considerations 8) Regulatory compliance issues"),
    
    ("How to measure AI ROI in business?", "AI ROI can be measured through: 1) Cost reduction metrics 2) Productivity improvements 3) Revenue increase 4) Customer satisfaction scores 5) Process efficiency gains 6) Error rate reduction 7) Time saved on tasks 8) Market share growth"),
    
    ("What is natural language processing in business?", "Natural Language Processing (NLP) in business enables computers to understand and process human language, powering applications like customer service chatbots, sentiment analysis, document processing, and automated report generation."),
    
    ("How does AI improve decision-making?", "AI improves business decision-making by analyzing vast amounts of data, identifying patterns, providing predictive insights, reducing human bias, and offering data-backed recommendations for strategic planning and risk management."),
    
    ("What industries benefit most from AI?", "Industries benefiting most from AI include: 1) Financial services 2) Healthcare 3) Retail 4) Manufacturing 5) Logistics 6) Technology 7) Marketing 8) Customer service. These sectors see significant improvements in efficiency and innovation."),
    
    ("What is AI automation in business?", "AI automation involves using intelligent systems to perform routine business tasks without human intervention, including data entry, document processing, customer support, inventory management, and financial transactions."),
    
    ("How to prepare workforce for AI?", "Prepare workforce for AI through: 1) Digital skills training 2) AI literacy programs 3) Change management 4) Technical workshops 5) Cross-functional training 6) Continuous learning initiatives 7) AI ethics education"),
    
    ("What is computer vision in business?", "Computer vision in business uses AI to analyze visual data, enabling applications like quality control in manufacturing, retail inventory management, security surveillance, and automated document processing."),
    
    ("How does AI enhance marketing?", "AI enhances marketing through personalized content delivery, customer behavior prediction, automated campaign optimization, real-time ad targeting, social media analysis, and customer journey mapping."),
    
    ("What is AI-driven analytics?", "AI-driven analytics combines artificial intelligence with data analysis to automatically discover insights, predict trends, identify anomalies, and generate recommendations for business optimization."),
    
    ("How to ensure AI ethics in business?", "Ensure AI ethics through: 1) Transparent AI policies 2) Bias testing 3) Regular audits 4) Privacy protection 5) Fair data practices 6) Ethical guidelines 7) Stakeholder involvement 8) Responsible AI development"),
    
    ("What's the future of AI in business?", "The future of AI in business includes: 1) Autonomous decision-making systems 2) Advanced predictive analytics 3) Hyper-personalization 4) Edge computing integration 5) AI-human collaboration 6) Quantum AI applications 7) Sustainable AI solutions"),

    ("What is AI-powered competitive analysis?", "AI-powered competitive analysis uses machine learning to monitor competitors' pricing, products, marketing strategies, and market positioning in real-time. It analyzes social media, websites, and market data to provide actionable insights about competitive advantages and market opportunities."),
    
    ("How does AI improve supply chain management?", "AI improves supply chain management through: 1) Demand forecasting 2) Inventory optimization 3) Route optimization 4) Supplier risk assessment 5) Real-time tracking 6) Quality control 7) Automated warehousing 8) Predictive maintenance of logistics equipment."),
    
    ("What is AI-driven customer segmentation?", "AI-driven customer segmentation uses machine learning algorithms to analyze customer data and automatically group customers based on behavior patterns, purchasing habits, demographics, and interactions. This enables highly targeted marketing and personalized service strategies."),
    
    ("How can AI detect financial fraud?", "AI detects financial fraud by analyzing patterns in transactions, identifying anomalies, monitoring user behavior, flagging suspicious activities, and using machine learning to adapt to new fraud patterns. It processes vast amounts of data in real-time to prevent fraudulent activities."),
    
    ("What is AI's role in business forecasting?", "AI in business forecasting uses historical data, market trends, external factors, and machine learning models to predict future business outcomes, including sales, demand, resource needs, market changes, and potential risks with higher accuracy than traditional methods."),
    
    ("How does AI transform HR processes?", "AI transforms HR through: 1) Automated candidate screening 2) Predictive hiring analytics 3) Employee performance assessment 4) Training personalization 5) Workforce planning 6) Employee engagement monitoring 7) Automated administrative tasks 8) Retention risk prediction."),
    
    ("What are AI-powered business intelligence tools?", "AI-powered BI tools combine traditional analytics with machine learning to provide: 1) Automated data analysis 2) Natural language querying 3) Predictive insights 4) Automated reporting 5) Real-time dashboards 6) Anomaly detection 7) Trend forecasting 8) Decision recommendations."),
    
    ("How does AI improve product development?", "AI enhances product development through: 1) Market trend analysis 2) Customer preference prediction 3) Design optimization 4) Prototype testing simulation 5) Quality control automation 6) Feature recommendation 7) Cost optimization 8) Development cycle acceleration."),
    
    ("What is AI's impact on pricing strategies?", "AI impacts pricing strategies through dynamic pricing algorithms that analyze market conditions, competitor prices, demand patterns, customer behavior, and other factors in real-time to optimize prices for maximum profit while maintaining market competitiveness."),
    
    ("How does AI enhance cybersecurity in business?", "AI enhances business cybersecurity by: 1) Detecting threats in real-time 2) Predicting potential vulnerabilities 3) Automating security responses 4) Analyzing network traffic patterns 5) Identifying unusual user behavior 6) Preventing data breaches 7) Managing access control 8) Updating security protocols automatically."),
    
    ("What is AI-driven risk management?", "AI-driven risk management uses machine learning to identify, assess, and mitigate business risks by analyzing historical data, market trends, and external factors. It provides real-time risk monitoring, automated alerts, and predictive risk scenarios."),
    
    ("How does AI improve business sustainability?", "AI improves business sustainability through: 1) Energy consumption optimization 2) Waste reduction analytics 3) Carbon footprint tracking 4) Sustainable supply chain management 5) Environmental impact prediction 6) Resource utilization optimization."),
    
    ("What is AI's role in compliance management?", "AI in compliance management automates regulatory monitoring, document analysis, policy enforcement, and reporting. It helps businesses stay compliant with changing regulations through real-time updates and automated compliance checking."),
    
    ("How does AI transform business meetings?", "AI transforms business meetings through: 1) Automated scheduling 2) Real-time translation 3) Meeting transcription 4) Action item extraction 5) Sentiment analysis 6) Participation monitoring 7) Automated follow-ups 8) Meeting effectiveness analytics."),
    
    ("What is AI's impact on business innovation?", "AI drives business innovation by: 1) Identifying market opportunities 2) Accelerating R&D processes 3) Enabling new business models 4) Automating creative processes 5) Facilitating rapid prototyping 6) Predicting innovation trends 7) Optimizing resource allocation for innovation projects.")
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

max_len = 75
dataset = create_dataset(conversation_pairs, word2idx, max_len)
print("Number of pairs:", len(dataset))
print("Sample 0:\n Input:", dataset[0][0], "\n Output:", dataset[0][1])

import math
import torch.nn as nn
import torch.nn.functional as F

class TransformerChat(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, num_layers=3, max_len=75):
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
batch_size = 8
chat_loader = DataLoader(chat_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerChat(vocab_size=len(vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx[PAD_TOKEN])

epochs = 250

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

def generate_reply(model, input_str, word2idx, idx2word, max_len=75):  

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

reply = generate_reply(model, "What is the role of Ai in business?", word2idx, idx2word, max_len = 40)
print("Bot says:", reply)

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    reply = generate_reply(model, user_input, word2idx, idx2word, max_len = 50)
    print("Bot:", reply)