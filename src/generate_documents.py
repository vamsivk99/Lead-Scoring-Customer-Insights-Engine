import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random

def generate_documents(num_documents=1000, output_dir="mock_docs"):
    """
    Generates a large number of detailed and varied synthetic financial documents
    using a powerful local model running on the user's GPU.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Check for GPU availability
    if not torch.cuda.is_available():
        raise SystemExit("GPU not available. This script is optimized for GPU usage.")
    
    try:
        # Step 1: Load the tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        print("Tokenizer loaded successfully.")

        # Step 2: Load the model
        print("Loading model onto GPU... This may take a moment.")
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype="auto",
            device_map="cuda", # More robust way to map to the GPU
            trust_remote_code=True
        )
        print("Model loaded successfully.")

        # Step 3: Create the generation pipeline from the loaded components
        print("Creating generation pipeline...")
        generator = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer
        )
        print("Pipeline created successfully.")

    except Exception as e:
        print(f"A critical error occurred during model loading: {e}")
        import traceback
        traceback.print_exc()
        return # Stop execution if loading fails
        
    # A much larger and more diverse list of prompts for document generation
    document_prompts = [
        "Create a detailed quarterly financial report for a publicly-traded tech company. Include a balance sheet, income statement, and cash flow statement. Add a section for Management's Discussion and Analysis (MD&A).",
        # Contracts and Agreements
        "Generate a detailed section for a commercial real estate loan agreement of $5,000,000, focusing on the 'Covenants' clause, detailing borrower responsibilities like property maintenance, insurance requirements, and financial reporting.",
        "Write a comprehensive 'Indemnification' clause for a software development contract between a startup and a Fortune 500 company. Cover liabilities related to intellectual property infringement, data breaches, and performance failures.",
        "Create a detailed 'Term and Termination' section for a 3-year exclusive distribution agreement for a new consumer electronics product in North America.",
        "Generate the 'Representations and Warranties' section for a stock purchase agreement where a larger company is acquiring a smaller tech startup for $50 million.",
        "Write a full, multi-page consulting agreement for a 6-month digital transformation project, including scope of work, deliverables, payment schedule ($50,000/month), and intellectual property rights.",
        "Create the user data privacy and security addendum for a SaaS agreement, ensuring compliance with both GDPR and CCPA regulations.",

        # Transcripts and Meeting Minutes
        "Write a 5-page transcript excerpt from a quarterly earnings call for a struggling retail company. The CEO should address declining sales, store closures, and the new e-commerce strategy. Include a tough question from a J.P. Morgan analyst.",
        "Generate the official meeting minutes for a company's board of directors meeting where they discuss and approve the annual budget. Include motions, votes, and abstentions.",
        "Create a transcript of a heated negotiation between a venture capital firm and a startup founder over the terms of a Series A funding round. Focus on the valuation and the board seat arguments.",
        "Write the transcript of a customer support call where a high-value client is complaining about a critical software bug that caused a business interruption.",
        
        # Financial Reports and Memos
        "Generate a detailed investment memorandum for a private equity firm considering a leveraged buyout of a mid-sized manufacturing company. Include sections on market analysis, competitive landscape, valuation analysis (DCF and Comps), and risk factors.",
        "Write the 'Management's Discussion and Analysis' (MD&A) section of an annual report for a pharmaceutical company that just received FDA approval for a blockbuster drug.",
        "Create a detailed credit analysis report for a bank considering a $10 million line of credit for a construction company. Analyze their balance sheet, income statement, and cash flow statement, and calculate key financial ratios.",
        "Generate a risk assessment report for a proposed investment in a volatile emerging market. Cover political risk, currency risk, and market risk with potential mitigation strategies.",
        "Write a due diligence summary report for an M&A transaction, highlighting key findings in the financial, legal, and operational review of the target company. Include a list of 'red flag' issues.",
        
        # Customer Communications
        "Write a formal letter from a bank to a corporate client, approving a significant increase in their credit line and outlining the new terms and conditions.",
        "Generate a detailed email chain between a financial advisor and a high-net-worth client discussing a major shift in their investment portfolio strategy due to changing market conditions.",
        "Create a proposal from a wealth management firm to a potential client, detailing their investment philosophy, services offered, and fee structure for a $15 million portfolio.",
        "Write a script for a sales call where a financial services provider is pitching a new treasury management solution to a corporate CFO.",
        "Generate a customer complaint letter regarding a complex billing error on a corporate account and the detailed, apologetic response from the service provider's customer success manager.",
        "Write a script for a 2-minute promotional video for a new sustainable investment fund. Highlight its key features and target audience."
    ]

    print(f"Generating {num_documents} high-quality financial documents using local GPU...")
    for i in tqdm(range(num_documents)):
        base_prompt = random.choice(document_prompts)
        # Phi-2 models follow a specific instruction format.
        prompt = f"Instruct: {base_prompt}\nOutput:"
        try:
            # The pipeline returns a list of dictionaries.
            generated_sequences = generator(
                prompt,
                max_new_tokens=1024,
                num_return_sequences=1,
                truncation=True,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )
            # The output text needs to be cleaned of the original prompt
            if generated_sequences and isinstance(generated_sequences, list) and 'generated_text' in generated_sequences[0]:
                full_text = generated_sequences[0]['generated_text']
                doc_content = full_text.split("Output:")[-1].strip()
            else:
                print(f"Warning: Received unexpected output format for document {i+1}. Skipping.")
                continue

            file_path = os.path.join(output_dir, f"document_{i+1}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(doc_content)
        except Exception as e:
            print(f"Error generating document {i+1}: {e}")
            continue

    print(f"\nSuccessfully generated {num_documents} documents in '{output_dir}'.")

if __name__ == "__main__":
    generate_documents(num_documents=1000) 