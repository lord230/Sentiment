# Created By LORD 
import torch

CSV_PATH    = "dataset/Domain.csv"
CSV_PATH_ABSA = "dataset/test_sentiment_SA.csv"
CSV_PATH_NEW = "dataset/combined_shuffled.csv"
CSV_PATH_SINN = "dataset/output.csv"
CSV_PATH_MANDIRA = "dataset/test_SINN.csv"
BATCH_SIZE  = 32
MAX_LEN     = 20
MAX_SAMPLES = 10000  # Set to None to use full dataset, or an int to limit dataset size
EPOCHS      = 5
LR          = 2e-4
SEED        = 42
CHECK_DIR   = "CheckPoints"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")