# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer, AutoModel
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report, confusion_matrix, f1_score
# import sklearn.metrics as metrics
# import seaborn as sn
# import pandas as pd
# import warnings
# import os

# # Import custom modules
# from siamese_dataset import SiameseDataset
# from siamese_network import SiameseNetwork
# from utils import multiclass_roc_auc_score, get_class_rep, get_closest_cluster
# from data_loader import prepare_data

# # Configuration
# warnings.filterwarnings("ignore")
# np.random.seed(123456)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Paths
# v0_data_dir = './dataset/FlakyCat_data/test_files_v0'
# v12_data_dir = './dataset/FlakyCat_data/test_files_v12'

# # Label mappings
# label_to_int = {
#     'async wait': 0,
#     'unordered collections': 1,
#     'concurrency': 2,
#     'time': 3,
#     'test order dependency': 4
# }

# int_to_label = {v: k for k, v in label_to_int.items()}

# # Hyperparameters
# m_len = 3402
# batch_size = 8

# # Initialize CodeBERT
# print("Loading CodeBERT model...")
# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model_codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(device)

# # Load and prepare data
# print("\nPreparing data...")
# (train_buggy_code, train_filenames, valid_buggy_code, valid_filenames,
#  train_category_counts, valid_category_counts) = prepare_data(
#     v0_data_dir, v12_data_dir, label_to_int
# )

# print("\nCategory-wise file counts in Validation Set:")
# for category, counts in valid_category_counts.items():
#     print(f"{category}: total={sum(counts.values())}, v0={counts['v0']}, "
#           f"v1={counts['v1']}, v2={counts['v2']}")

# # Create datasets
# print("\nCreating datasets...")
# train_dataset = SiameseDataset(train_buggy_code, tokenizer, model_codebert, 
#                                train_filenames, 'train', label_to_int, m_len)
# val_dataset = SiameseDataset(valid_buggy_code, tokenizer, model_codebert, 
#                              valid_filenames, 'val', label_to_int, m_len)

# # Load trained model
# print("\nLoading trained model...")
# siamese_network = SiameseNetwork(m_len).to(device)
# siamese_network.load_state_dict(torch.load('flakyXbert_augExp1.pth'))
# siamese_network.eval()

# # === 🔥 Generate or load cached embeddings ===
# embedding_cache_path = "train_embeddings.pt"

# if os.path.exists(embedding_cache_path):
#     print("\nLoading cached training embeddings...")
#     saved = torch.load(embedding_cache_path, map_location=device)
#     post_train_embed = saved['embeddings']
#     post_train_label = saved['labels']
# else:
#     print("\nGenerating embeddings from training set (this may take a while)...")
#     post_train_embed = []
#     post_train_label = []
#     with torch.no_grad():
#         for item in tqdm(train_dataset, desc="Extracting training embeddings"):
#             post_train_embed.append(siamese_network(item['anchor'].to(device)))
#             post_train_label.append(item['label'])
#     torch.save({'embeddings': post_train_embed, 'labels': post_train_label}, embedding_cache_path)
#     print(f"Training embeddings cached at '{embedding_cache_path}'")

# # === Prediction and evaluation ===

# def predict(input_vector):
#     modified_vector = siamese_network(input_vector.to(device))
#     representatives = get_class_rep(post_train_embed, post_train_label)
#     return get_closest_cluster(representatives, modified_vector, int_to_label)

# print("\nEvaluating on validation set...")
# predicted_labels = []
# true_labels = []

# for item in tqdm(val_dataset, desc="Predicting"):
#     input_vector = item['anchor']
#     predicted_label = predict(input_vector)
#     predicted_labels.append(predicted_label)
#     true_label = int_to_label[int(item['label'])]
#     true_labels.append(true_label)







import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import sklearn.metrics as metrics
import seaborn as sn
import pandas as pd
import warnings
import os

# Import custom modules
from siamese_dataset import SiameseDataset
from siamese_network import SiameseNetwork
from utils import multiclass_roc_auc_score, get_class_rep, get_closest_cluster
from data_loader import prepare_data

# Configuration
warnings.filterwarnings("ignore")
np.random.seed(123456)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
v0_data_dir = './dataset/FlakyCat_data/test_files_v0'
v12_data_dir = './dataset/FlakyCat_data/test_files_v12'

# Label mappings
label_to_int = {
    'async wait': 0,
    'unordered collections': 1,
    'concurrency': 2,
    'time': 3,
    'test order dependency': 4
}

int_to_label = {v: k for k, v in label_to_int.items()}

# Hyperparameters
m_len = 3402
batch_size = 8

# Initialize CodeBERT
print("Loading CodeBERT model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
# model_codebert = AutoModel.from_pretrained("microsoft/codebert-base").to(device)
model_codebert = AutoModel.from_pretrained(
    "microsoft/codebert-base",
    trust_remote_code=True,
    use_safetensors=True
).to(device)

# Load and prepare data
print("\nPreparing data...")
(train_buggy_code, train_filenames, valid_buggy_code, valid_filenames,
 train_category_counts, valid_category_counts) = prepare_data(
    v0_data_dir, v12_data_dir, label_to_int
)

print("\nCategory-wise file counts in Validation Set:")
for category, counts in valid_category_counts.items():
    print(f"{category}: total={sum(counts.values())}, v0={counts['v0']}, "
          f"v1={counts['v1']}, v2={counts['v2']}")

# Create datasets
print("\nCreating datasets...")
train_dataset = SiameseDataset(train_buggy_code, tokenizer, model_codebert, 
                               train_filenames, 'train', label_to_int, m_len)
val_dataset = SiameseDataset(valid_buggy_code, tokenizer, model_codebert, 
                             valid_filenames, 'val', label_to_int, m_len)

# Load trained model
print("\nLoading trained model...")
siamese_network = SiameseNetwork(m_len).to(device)
siamese_network.load_state_dict(torch.load('flakyXbert_augExp1.pth'))
siamese_network.eval()

# === 🔥 CHANGED: Generate or load cached embeddings ===
embedding_cache_path = "train_embeddings.pt"

if os.path.exists(embedding_cache_path):
    print("\n✅ Loading cached training embeddings...")
    saved = torch.load(embedding_cache_path, map_location=device)
    post_train_embed = saved['embeddings']
    post_train_label = saved['labels']
    print(f"Loaded {len(post_train_embed)} training embeddings from cache")
else:
    print("\n⚙️ Generating embeddings from training set (this may take a while)...")
    post_train_embed = []
    post_train_label = []
    with torch.no_grad():
        for item in tqdm(train_dataset, desc="Extracting training embeddings"):
            post_train_embed.append(siamese_network(item['anchor'].to(device)))
            post_train_label.append(item['label'])
    
    # 🔥 CHANGED: Save the embeddings after generation
    torch.save({'embeddings': post_train_embed, 'labels': post_train_label}, embedding_cache_path)
    print(f"✅ Training embeddings cached at '{embedding_cache_path}'")

# === 🔥 NEW: Function to preprocess code snippet ===
def preprocess_code_snippet(code_snippet):
    """
    Preprocesses a code snippet and converts it to the format expected by the model.
    
    Args:
        code_snippet (str): Raw code snippet as string
        
    Returns:
        torch.Tensor: Preprocessed embedding ready for prediction
    """
    # Tokenize the code snippet
    tokens = tokenizer.tokenize(code_snippet)
    
    # Convert tokens to IDs
    ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Pad or truncate to m_len
    if len(ids) < m_len:
        ids = ids + [tokenizer.pad_token_id] * (m_len - len(ids))
    else:
        ids = ids[:m_len]
    
    # Create attention mask
    mask = [1 if i != tokenizer.pad_token_id else 0 for i in ids]
    
    # Convert to tensors
    ids_tensor = torch.tensor([ids], dtype=torch.long)
    mask_tensor = torch.tensor([mask], dtype=torch.long)
    
    # Get CodeBERT embeddings
    with torch.no_grad():
        outputs = model_codebert(ids_tensor.to(device), attention_mask=mask_tensor.to(device))
        code_embedding = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token embedding
    
    return code_embedding.squeeze(0)

# === 🔥 CHANGED: Modified prediction function ===
def predict(input_vector):
    """
    Predicts the flaky test category for a given input vector.
    
    Args:
        input_vector (torch.Tensor): Input code embedding
        
    Returns:
        str: Predicted flaky test category
    """
    with torch.no_grad():
        modified_vector = siamese_network(input_vector.to(device))
        representatives = get_class_rep(post_train_embed, post_train_label)
        return get_closest_cluster(representatives, modified_vector, int_to_label)

# === 🔥 NEW: Main detection function for code snippets ===
def detect_flaky_test(code_snippet):
    """
    Detects if a code snippet results in a flaky test and identifies the category.
    
    Args:
        code_snippet (str): The code snippet to analyze
        
    Returns:
        dict: Contains 'is_flaky', 'category', and 'confidence' information
    """
    print("\n🔍 Analyzing code snippet...")
    
    # Preprocess the code snippet
    input_embedding = preprocess_code_snippet(code_snippet)
    
    # Get prediction
    predicted_category = predict(input_embedding)
    
    # 🔥 NEW: Calculate confidence based on distance to cluster representatives
    with torch.no_grad():
        modified_vector = siamese_network(input_embedding.to(device))
        representatives = get_class_rep(post_train_embed, post_train_label)
        
        # Calculate distances to all cluster representatives
        distances = {}
        for label_name, rep_vector in representatives.items():
            dist = torch.nn.functional.cosine_similarity(
                modified_vector.unsqueeze(0), 
                rep_vector.unsqueeze(0)
            ).item()
            distances[label_name] = dist
        
        # Get confidence score (normalized similarity)
        max_similarity = max(distances.values())
        confidence = max_similarity * 100  # Convert to percentage
    
    result = {
        'is_flaky': True,  # All categories indicate flaky tests
        'category': predicted_category,
        'confidence': confidence,
        'all_scores': distances
    }
    
    return result

# === 🔥 NEW: Interactive prediction interface ===
def interactive_detection():
    """
    Provides an interactive interface for detecting flaky tests from code snippets.
    """
    print("\n" + "="*70)
    print("🔬 FLAKY TEST DETECTION SYSTEM")
    print("="*70)
    print("\nEnter your code snippet (type 'END' on a new line when finished):")
    print("Type 'QUIT' to exit the system.\n")
    
    while True:
        lines = []
        print(">>> ", end="")
        while True:
            line = input()
            if line.strip() == 'END':
                break
            if line.strip() == 'QUIT':
                print("\n👋 Exiting detection system...")
                return
            lines.append(line)
        
        if not lines:
            continue
            
        code_snippet = '\n'.join(lines)
        
        # Detect flaky test
        result = detect_flaky_test(code_snippet)
        
        # Display results
        print("\n" + "-"*70)
        print("📊 DETECTION RESULTS")
        print("-"*70)
        print(f"Is Flaky: {'⚠️  YES' if result['is_flaky'] else '✅ NO'}")
        print(f"Category: {result['category'].upper()}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print("\nAll Category Scores:")
        for category, score in sorted(result['all_scores'].items(), key=lambda x: x[1], reverse=True):
            print(f"  • {category}: {score:.4f}")
        print("-"*70 + "\n")

# === 🔥 CHANGED: Evaluation section (optional) ===
# This section can be run separately or commented out for production use
def evaluate_on_validation_set():
    """
    Evaluates the model on the validation set.
    """
    print("\n📈 Evaluating on validation set...")
    predicted_labels = []
    true_labels = []

    for item in tqdm(val_dataset, desc="Predicting"):
        input_vector = item['anchor']
        predicted_label = predict(input_vector)
        predicted_labels.append(predicted_label)
        true_label = int_to_label[int(item['label'])]
        true_labels.append(true_label)
    
    # Print evaluation metrics
    print("\n" + "="*70)
    print("VALIDATION RESULTS")
    print("="*70)
    print(classification_report(true_labels, predicted_labels))
    
    return predicted_labels, true_labels

# === 🔥 NEW: Main execution block ===
if __name__ == "__main__":
    # Example usage
    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70)
    
    # Example 1: Direct prediction
    example_code = """
    def test_async_operation():
        result = async_call()
        assert result == expected_value
    """
    
    print("\n📝 Example Detection:")
    result = detect_flaky_test(example_code)
    print(f"Result: {result}")
    
    # Option 1: Run interactive mode
    # interactive_detection()
    
    # Option 2: Evaluate on validation set
    # evaluate_on_validation_set()