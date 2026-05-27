import os

tasks = ["kcat", "km", "ki"]
modalities = ["seq_only", "bimodal", "trimodal"]

print("==================================================")
# Evaluating the fully convergent 2026 multimodal enzyme benchmark
print("🏆 STARTING BATCH INFERENCE ON DISJOINT OOD TEST SET (2026)")
print("==================================================")

for task in tasks:
    for mode in modalities:
        # Programmatically modify the control panel of evaluate.py
        with open("evaluate.py", "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if 'TARGET_TASK = ' in line:
                lines[i] = f'    TARGET_TASK = "{task}"       # Target choice. Options: "kcat", "km", "ki"\n'
            elif 'MODEL_MODE  = ' in line:
                lines[i] = f'    MODEL_MODE  = "{mode}"    # Modality profile. Options: "seq_only", "bimodal", "trimodal"\n'
                
        with open("evaluate.py", "w", encoding="utf-8") as f:
            f.writelines(lines)
            
        # Run the standard evaluation matrix cell
        os.system("python evaluate.py")

print("==================================================")
print("🎉 ALL TEST PERFORMANCE METRICS SUCCESSFULLY EXTRACTED!")
print("==================================================")