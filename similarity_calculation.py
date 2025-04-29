# ================================
# Import Libraries
# ================================
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ================================
# Configuration
# ================================
CSV_PATH = 'DataNeuron_Text_Similarity.csv'
OUTPUT_PATH = 'DataNeuron_Text_Similarity_With_Scores.csv'
MODEL_NAME = 'paraphrase-MiniLM-L3-v2'  # Lightweight model good for semantic similarity


# ================================
# Load and Verify Dataset
# ================================
def load_and_verify_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("\nData loaded successfully. First few rows:")
        print(df.head())
        
        # Check for required columns
        required_columns = []
        for col in ['text1', 'text2', 'paragraph1', 'paragraph2', 'para1', 'para2']:
            if col in df.columns:
                required_columns = [col.replace('1', '').replace('2', '') + '1', 
                                  col.replace('1', '').replace('2', '') + '2']
                break
        
        if not required_columns:
            raise ValueError("Could not find any recognized column names in the CSV.")
            
        print(f"\nUsing columns: {required_columns[0]} and {required_columns[1]}")
        return df, required_columns[0], required_columns[1]
    
    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        print("Please ensure:")
        print("1. The file exists at the specified path")
        print("2. The file is a valid CSV")
        print("3. It contains at least two text columns")
        print("\nActual columns found (if any):")
        try:
            print(df.columns.tolist())
        except:
            pass
        exit()

# ================================
# Similarity Calculation with Robust Error Handling
# ================================
class SemanticSimilarityCalculator:
    def __init__(self, model_name=MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        
    def calculate_similarity(self, text1, text2):
        try:
            if pd.isna(text1) or pd.isna(text2) or text1.strip() == "" or text2.strip() == "":
                return 0.0
                
            embeddings = self.model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return round(float(similarity), 4)
        except Exception as e:
            print(f"Error processing texts: {str(e)}")
            return 0.0

# ================================
# Main Execution
# ================================
if __name__ == "__main__":
    print("Starting semantic similarity calculation...")
    
    # Load data with verification
    df, col1, col2 = load_and_verify_data(CSV_PATH)
    
    # Initialize calculator
    calculator = SemanticSimilarityCalculator()
    
    # Calculate similarity scores
    print("\nCalculating similarity scores...")
    df['similarity_score'] = df.apply(
        lambda row: calculator.calculate_similarity(row[col1], row[col2]), 
        axis=1
    )
    
    # Save results
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nCalculation complete! Results saved to {OUTPUT_PATH}")
    print("\nSample results:")
    print(df[[col1, col2, 'similarity_score']].head())