"""
main.py
AI-Powered Medical Image Analysis System - Master Execution Pipeline.
Orchestrates:
  1. Data discovery / Synthetic clinical radiograph generation
  2. Medical image preprocessing & clinical augmentation
  3. Deep Learning model initialization
  4. Model training, validation, and checkpointing
  5. Comprehensive clinical evaluation (Sensitivity, Specificity, AUC)
  6. Visual asset generation (Curves, Confusion Matrix, Saliency Heatmaps)
  7. Virtual Clinical PACS Emergency Triage Simulation
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image

from src.data_loader import load_dataset, train_val_test_split, setup_synthetic_dataset
from src.preprocessor import MedicalImagePreprocessor
from src.model import MedicalCNN
from src.trainer import MedicalModelTrainer
from src.evaluator import ClinicalEvaluator
from src.visualizer import ClinicalVisualizer
from src.simulation import VirtualHospitalSimulator


if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def run_pipeline(epochs=20, batch_size=16, base_dir="data", output_dir="outputs", image_dir="images"):
    print("=" * 75)
    print("   [+] AI-POWERED MEDICAL IMAGE ANALYSIS & CLINICAL TRIAGE SYSTEM")
    print("=" * 75)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Dataset Ingestion & Discovery
    # -------------------------------------------------------------
    print("\n[Phase 1/7] Discovering thoracic radiography dataset...")
    images, labels, file_paths = load_dataset(base_dir=base_dir, target_size=(128, 128))
    print(f" -> Total radiographs loaded: {len(labels)} (Normal: {np.sum(labels==0)}, Pneumonia: {np.sum(labels==1)})")

    # Train / Val / Test Partitioning
    split = train_val_test_split(images, labels, file_paths=file_paths, val_ratio=0.15, test_ratio=0.15, seed=42)
    X_train, y_train = split["X_train"], split["y_train"]
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]
    paths_test = split["paths_test"]

    print(f" -> Train cohort: {len(y_train)} | Validation: {len(y_val)} | Test: {len(y_test)}")

    # -------------------------------------------------------------
    # Step 2: Clinical Preprocessing & Augmentation
    # -------------------------------------------------------------
    print("\n[Phase 2/7] Applying contrast enhancement (CLAHE) & clinical augmentation...")
    preprocessor = MedicalImagePreprocessor(target_size=(128, 128), enable_clahe=True)

    # Augment training set to simulate diverse radiographic exposure
    X_train_aug, y_train_aug = preprocessor.augment_batch(X_train, y_train, augment_factor=1)
    print(f" -> Augmented training cohort size: {len(y_train_aug)} images")

    # -------------------------------------------------------------
    # Step 3: Model Architecture Initialization
    # -------------------------------------------------------------
    print("\n[Phase 3/7] Building Medical CNN Architecture...")
    model = MedicalCNN(input_shape=(128, 128, 1), seed=42)
    print(f" -> Model initialized with {model.num_spatial_features} spatial feature descriptors.")
    print(f" -> Activation: ReLU (hidden), Sigmoid (output probability).")

    # -------------------------------------------------------------
    # Step 4: Model Training Loop & Checkpointing
    # -------------------------------------------------------------
    print("\n[Phase 4/7] Initiating training loop with backpropagation & momentum...")
    trainer = MedicalModelTrainer(model, learning_rate=0.035, momentum=0.88, l2_reg=0.0005)
    history = trainer.fit(X_train_aug, y_train_aug, X_val, y_val, epochs=epochs, batch_size=batch_size, verbose=True)

    # Save model weights & metadata
    weights_path = os.path.join("models", "model_weights.npy")
    meta_path = os.path.join("models", "model_metadata.json")
    model.save(weights_path, meta_path)
    print(f" -> Checkpoint saved to: {weights_path}")

    # -------------------------------------------------------------
    # Step 5: Clinical Evaluation & Metrics
    # -------------------------------------------------------------
    print("\n[Phase 5/7] Evaluating diagnostic performance on held-out test cohort...")
    test_preds, test_probs = model.predict(X_test, threshold=0.5)

    evaluator = ClinicalEvaluator(y_test, test_probs, threshold=0.5)
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    metrics = evaluator.save_report(metrics_path)

    print(f" -> Diagnostic Accuracy:      {metrics['accuracy']*100:.2f}%")
    print(f" -> Sensitivity (Recall):    {metrics['sensitivity_recall']*100:.2f}%")
    print(f" -> Specificity:             {metrics['specificity']*100:.2f}%")
    print(f" -> Precision (PPV):         {metrics['precision_ppv']*100:.2f}%")
    print(f" -> Area Under Curve (AUC):  {metrics['roc_auc']:.4f}")
    print(f" -> Optimal Cutoff:          {metrics['optimal_clinical_threshold']:.4f}")

    # -------------------------------------------------------------
    # Step 6: Visualizations & Explainable AI (XAI)
    # -------------------------------------------------------------
    print("\n[Phase 6/7] Generating publication-grade clinical visualizations...")
    visualizer = ClinicalVisualizer(output_dir=image_dir)

    # 1. Preprocessing comparison
    raw_sample = X_test[0, ..., 0]
    proc_sample = preprocessor.preprocess_image(raw_sample)
    visualizer.plot_preprocessing_comparison(raw_sample, proc_sample, "preprocessing_comparison.png")

    # 2. Training history curves
    visualizer.plot_training_history(history, "training_history.png")

    # 3. Confusion Matrix
    visualizer.plot_confusion_matrix(y_test, test_preds, "confusion_matrix.png")

    # 4. ROC Curve
    visualizer.plot_roc_curve(y_test, test_probs, "roc_curve.png")

    # 5. Explainable AI Grad-CAM Saliency Map
    # Select a true pneumonia sample if available
    pneumonia_indices = np.where(y_test == 1)[0]
    xai_idx = pneumonia_indices[0] if len(pneumonia_indices) > 0 else 0
    xai_img = X_test[xai_idx]
    xai_heatmap = model.generate_saliency_heatmap(xai_img)
    visualizer.plot_explainable_gradcam(
        xai_img,
        xai_heatmap,
        label_text="PNEUMONIA" if test_preds[xai_idx] == 1 else "NORMAL",
        prob_val=float(test_probs[xai_idx]),
        save_name="gradcam_saliency_map.png"
    )

    # 6. Save sample benchmark images for documentation
    norm_sample_pil = Image.fromarray((X_test[np.where(y_test == 0)[0][0], ..., 0] * 255).astype(np.uint8))
    norm_sample_pil.save(os.path.join(image_dir, "sample_xray_normal.png"))

    pneu_sample_pil = Image.fromarray((X_test[xai_idx, ..., 0] * 255).astype(np.uint8))
    pneu_sample_pil.save(os.path.join(image_dir, "sample_xray_pneumonia.png"))

    # -------------------------------------------------------------
    # Step 7: Virtual Clinical PACS Emergency Triage Simulation
    # -------------------------------------------------------------
    print("\n[Phase 7/7] Running Virtual Hospital PACS Emergency Triage Simulation...")
    simulator = VirtualHospitalSimulator(model, preprocessor)

    # Build simulated patient study records for test cohort
    patient_records = []
    for i in range(len(y_test)):
        fpath = paths_test[i] if i < len(paths_test) else f"data/raw/{'pneumonia' if y_test[i]==1 else 'normal'}/sample.png"
        patient_records.append({
            "patient_id": f"PT-ER-{202600 + i}",
            "filepath": fpath,
            "label": "PNEUMONIA" if y_test[i] == 1 else "NORMAL"
        })

    report_path = os.path.join(output_dir, "clinical_triage_report.json")
    triage_summary = simulator.simulate_emergency_worklist(patient_records, output_report_path=report_path)

    # Generate batch prediction dataframe
    batch_records = []
    triage_plot_samples = []

    for item in triage_summary["prioritized_patient_worklist"]:
        batch_records.append({
            "Patient_ID": item["patient_id"],
            "Ground_Truth": item["actual_ground_truth"],
            "AI_Prediction": item["ai_predicted_class"],
            "Pneumonia_Probability": item["pneumonia_probability"],
            "Triage_Urgency": item["triage_urgency_tier"],
            "Clinical_Action": item["clinical_action_recommendation"],
            "Inference_Latency_ms": item["simulated_inference_latency_ms"]
        })

    df_batch = pd.DataFrame(batch_records)
    csv_path = os.path.join(output_dir, "batch_predictions.csv")
    df_batch.to_csv(csv_path, index=False)

    # Generate triage visual dashboard
    for i in range(min(6, len(y_test))):
        triage_plot_samples.append({
            "image": X_test[i],
            "patient_id": patient_records[i]["patient_id"],
            "ground_truth": "PNEUMONIA" if y_test[i] == 1 else "NORMAL",
            "pred": "PNEUMONIA" if test_preds[i] == 1 else "NORMAL",
            "prob": float(test_probs[i]),
            "triage": triage_summary["prioritized_patient_worklist"][i]["triage_urgency_tier"]
        })

    visualizer.plot_triage_dashboard(triage_plot_samples, "clinical_triage_dashboard.png")

    # Also copy all generated images into outputs/ for easy browsing
    for img_file in os.listdir(image_dir):
        if img_file.endswith(".png"):
            src = os.path.join(image_dir, img_file)
            dst = os.path.join(output_dir, img_file)
            with open(src, "rb") as sf, open(dst, "wb") as df:
                df.write(sf.read())

    print("\n" + "=" * 75)
    print("   [SUCCESS] PIPELINE EXECUTION COMPLETE - SUMMARY OF GENERATED ASSETS")
    print("=" * 75)
    print(f" [DIR] Models Directory:   models/model_weights.npy, models/model_metadata.json")
    print(f" [METRICS] Metrics File:   outputs/evaluation_metrics.json")
    print(f" [REPORT] Triage Report:   outputs/clinical_triage_report.json")
    print(f" [CSV] Batch CSV:          outputs/batch_predictions.csv")
    print(f" [FIGURES] Images:         images/ (6 high-res plots + sample radiographs)")
    print(f" [WEB APP] Interactive UI: Run 'python app.py' to launch local PACS portal")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Powered Medical Image Analysis System")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    args = parser.parse_args()

    run_pipeline(epochs=args.epochs, batch_size=args.batch_size)
