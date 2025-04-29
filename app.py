# ================================
# Import Libraries
# ================================
from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
# ================================
# Initialize Flask & Model
# ================================
app = Flask(__name__)
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Much smaller model

# ================================
# Similarity Calculation Function
# ================================
def calculate_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(similarity), 4)

# ================================
# Root Endpoint
# ================================
@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <head>
            <title>Text Similarity API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: #333; }
                .container { border: 1px solid #ddd; padding: 20px; border-radius: 5px; }
                label { display: block; margin-top: 10px; }
                textarea { width: 100%; height: 100px; margin-bottom: 10px; padding: 8px; }
                button { background: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }
                #result { margin-top: 20px; padding: 10px; background: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>Text Similarity Calculator</h1>
            <div class="container">
                <label for="text1">Text 1:</label>
                <textarea id="text1" placeholder="Enter first text here"></textarea>
                
                <label for="text2">Text 2:</label>
                <textarea id="text2" placeholder="Enter second text here"></textarea>
                
                <button onclick="calculateSimilarity()">Calculate Similarity</button>
                
                <div id="result"></div>
            </div>
            
            <script>
                function calculateSimilarity() {
                    const text1 = document.getElementById('text1').value;
                    const text2 = document.getElementById('text2').value;
                    
                    if (!text1 || !text2) {
                        document.getElementById('result').innerHTML = 'Please enter both texts';
                        return;
                    }
                    
                    fetch('/similarity', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text1, text2 })
                    })
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('result').innerHTML = 
                            `<strong>Similarity Score:</strong> ${data['similarity score']}`;
                    })
                    .catch(error => {
                        document.getElementById('result').innerHTML = 'Error calculating similarity';
                        console.error(error);
                    });
                }
            </script>
        </body>
    </html>
    """

# ================================
# API Endpoint
# ================================
@app.route('/similarity', methods=['POST'])
def similarity():
    try:
        data = request.json
        
        # Validate input format
        if 'text1' not in data or 'text2' not in data:
            return jsonify({"error": "Request must contain 'text1' and 'text2'"}), 400
        
        # Calculate similarity
        score = calculate_similarity(data['text1'], data['text2'])
        
        # Return response in required format
        return jsonify({"similarity score": score})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================
# Run Flask App (Local Testing)
# ================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)