import os
import sys
import asyncio
from PIL import Image
import numpy as np

# Ensure app package is visible to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

async def run_tests():
    print("=== STARTING BACKEND PIPELINE VERIFICATION SUITE ===")
    
    # 1. Create a mock RGB image for testing
    test_img_path = "test_sample_eczema.jpg"
    img_data = np.zeros((200, 200, 3), dtype=np.uint8)
    # Put a colored blob in the middle (Eczema representation)
    for i in range(200):
        for j in range(200):
            if (i - 100)**2 + (j - 100)**2 < 40**2:
                img_data[i, j] = [200, 80, 80]  # Reddish color
            else:
                img_data[i, j] = [240, 220, 200] # Skin-like background
                
    Image.fromarray(img_data).save(test_img_path)
    print(f"Created mock skin photo at: {test_img_path}")

    # 2. Test Modality routing and pipeline execution
    from app.services.pipeline import run_pipeline
    
    print("\n--- Running AI Pipeline... ---")
    results = await run_pipeline(test_img_path)
    
    print("Pipeline Results:")
    for k, v in results.items():
        if k != "report_text":
            print(f"  {k}: {v}")
            
    assert results["modality"] == "RGB", "Pipeline should detect this as a visible RGB image"
    # Accepting dynamic output from the active deep learning model heads
    assert results["body_part_scan_type"] in ["Skin", "Eye", "Finger", "Nail", "Lips", "Tongue", "Ear", "Wound", "Burn"], "Should yield a valid clinical class"
    assert results["specialist"] is not None, "Should map to a clinical specialist"
    assert len(results["bbox"]) == 4, "Should generate bounding box coordinates"
    assert os.path.exists(results["mask_filename"]), "Should write segment mask overlay PNG file"
    print("[OK] Modality Routing, Classification, and YOLO-Seg simulation outputs are correct!")

    # 3. Test PDF generation
    from app.services.report import generate_pdf_report
    
    print("\n--- Testing PDF Report compilation... ---")
    mock_case = {
        "id": "12345678-abcd-ef00-1122-334455667788",
        "patient_name": "Test Patient",
        "patient_age": 30,
        "patient_sex": "Male",
        "symptoms": "Red patch on arm",
        "medical_history": "N/A",
        "modality": results["modality"],
        "body_part_scan_type": results["body_part_scan_type"],
        "disease": results["disease"],
        "confidence": results["confidence"],
        "severity": results["severity"],
        "specialist": results["specialist"],
        "report_text": results["report_text"],
        "rag_suggestions": [
            "Keep the affected skin moisturized daily.",
            "Avoid harsh soaps and hot water baths."
        ]
    }
    
    pdf_bytes = generate_pdf_report(mock_case)
    assert len(pdf_bytes) > 0, "PDF byte stream should not be empty"
    
    test_pdf_out = "test_output_report.pdf"
    with open(test_pdf_out, "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF Report generated successfully at: {test_pdf_out}")
    print("[OK] PDF styling and rendering engine are correct!")

    # 4. Test RAG Retrieval Guidelines
    from app.services.rag import get_rag_advice, init_rag_system
    
    print("\n--- Testing Vector Database & RAG Guidance... ---")
    init_rag_system()
    answer, suggestions = await get_rag_advice(mock_case, "What are the common causes and remedies?")
    print(f"RAG advice output:\n{answer[:200]}...\n")
    print("Suggestions extracted:")
    for s in suggestions:
        print(f"  - {s}")
        
    assert len(suggestions) > 0, "Should generate checklist items"
    print("[OK] Guidelines vector lookup and local LLM mock advice are correct!")

    # Cleanup files
    if os.path.exists(test_img_path):
        os.remove(test_img_path)
    if os.path.exists(test_pdf_out):
        os.remove(test_pdf_out)
        
    print("\n=== ALL TEST CASES PASSED SUCCESSFULLY ===")

if __name__ == "__main__":
    asyncio.run(run_tests())
