from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
import pandas as pd
import os

def create_presentation():
    prs = Presentation()
    
    # --- Slide Layouts ---
    # 0: Title, 1: Title and Content, 2: Section Header, 3: Two Content, 4: Comparison, 5: Title Only, 6: Blank
    TITLE_SLIDE_LAYOUT = prs.slide_layouts[0]
    CONTENT_SLIDE_LAYOUT = prs.slide_layouts[1]
    TWO_CONTENT_LAYOUT = prs.slide_layouts[3]
    
    # ---------------------------------------------------------
    # Slide 1: Title Slide
    # ---------------------------------------------------------
    slide1 = prs.slides.add_slide(TITLE_SLIDE_LAYOUT)
    title = slide1.shapes.title
    subtitle = slide1.placeholders[1]
    
    title.text = "Modernizing YOLO-LITE"
    subtitle.text = "An Ablation Study on Advanced Activation Functions\nUsing PyTorch"
    
    # Style Title
    title.text_frame.paragraphs[0].font.size = Pt(44)
    title.text_frame.paragraphs[0].font.bold = True
    title.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 51, 102)

    # ---------------------------------------------------------
    # Slide 2: The Original YOLO-LITE Architecture
    # ---------------------------------------------------------
    slide2 = prs.slides.add_slide(CONTENT_SLIDE_LAYOUT)
    slide2.shapes.title.text = "The Original YOLO-LITE Architecture"
    
    tf = slide2.placeholders[1].text_frame
    tf.text = "Designed for Non-GPU Computers"
    
    p = tf.add_paragraph()
    p.text = "Goal: Achieve real-time object detection (~10 FPS) without a GPU."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Original Implementation: Written in C/C++ using the Darknet framework."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Baseline Metrics: Achieved 33.57% mAP on PASCAL VOC at 21 FPS (Dell XPS 13)."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Limitations: Relied entirely on the LeakyReLU activation function. Hard to prototype or extend with modern Deep Learning advances."
    p.level = 1

    # ---------------------------------------------------------
    # Slide 3: Motivation for PyTorch Transition
    # ---------------------------------------------------------
    slide3 = prs.slides.add_slide(CONTENT_SLIDE_LAYOUT)
    slide3.shapes.title.text = "Motivation: Why Python & PyTorch?"
    
    tf = slide3.placeholders[1].text_frame
    tf.text = "Overcoming Darknet's Constraints"
    
    p = tf.add_paragraph()
    p.text = "Rapid Experimentation: Python allows for seamless swapping of layers, loss functions, and activation functions."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Low-VRAM Optimization: Leveraged PyTorch Automatic Mixed Precision (AMP) and Gradient Accumulation to train on limited hardware."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Advanced Activations: Native support for modern, smooth, non-monotonic functions (e.g., SiLU, Mish) to improve gradient flow in shallow networks."
    p.level = 1

    # ---------------------------------------------------------
    # Slide 4: Theory: Advanced Activation Functions
    # ---------------------------------------------------------
    slide4 = prs.slides.add_slide(CONTENT_SLIDE_LAYOUT)
    slide4.shapes.title.text = "Theory: Activation Functions Analyzed"
    
    tf = slide4.placeholders[1].text_frame
    tf.text = "LeakyReLU (Baseline): Solves dying ReLU; f(x) = max(0.1x, x)"
    
    p = tf.add_paragraph()
    p.text = "ReLU: Fast, sparse, but suffers from zero gradients for negatives."
    p.level = 0
    
    p = tf.add_paragraph()
    p.text = "SiLU (Swish): f(x) = x * sigmoid(x). Smooth and non-monotonic, helping gradient flow, but requires exponential calculation."
    p.level = 0
    
    p = tf.add_paragraph()
    p.text = "Mish: f(x) = x * tanh(ln(1 + e^x)). Extremely smooth, often yields higher accuracy, but is computationally heavier."
    p.level = 0
    
    p = tf.add_paragraph()
    p.text = "HardSwish: Piecewise linear approximation of SiLU. Designed for fast mobile/edge deployment."
    p.level = 0

    # ---------------------------------------------------------
    # Slide 5: Experimental Setup
    # ---------------------------------------------------------
    slide5 = prs.slides.add_slide(CONTENT_SLIDE_LAYOUT)
    slide5.shapes.title.text = "Experimental Setup"
    
    tf = slide5.placeholders[1].text_frame
    tf.text = "Dataset: Pascal VOC 2007 + 2012"
    
    p = tf.add_paragraph()
    p.text = "Network: YOLO-LITE 'trial13' architecture (8 Convolutional Layers)."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Optimizer: SGD (Learning Rate = 0.001, Momentum = 0.9)."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Hardware Optimization: Mixed Precision (FP16), Batch Size 8 (Effective batch size 32 via Gradient Accumulation)."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Metrics Tracked: Validation Loss, mAP@0.5, CPU FPS, GPU FPS, Model Size."
    p.level = 1

    # ---------------------------------------------------------
    # Slide 6: Visual Results - Activation Shapes
    # ---------------------------------------------------------
    slide6 = prs.slides.add_slide(TITLE_SLIDE_LAYOUT)
    slide6.shapes.title.text = "Visualizing Activation Shapes"
    slide6.placeholders[1].text = "" # Clear subtitle
    
    img_path = "logs/plots/activation_functions.png"
    if os.path.exists(img_path):
        slide6.shapes.add_picture(img_path, Inches(1.5), Inches(2.5), width=Inches(7))

    # ---------------------------------------------------------
    # Slide 7: Visual Results - Validation Loss
    # ---------------------------------------------------------
    slide7 = prs.slides.add_slide(TITLE_SLIDE_LAYOUT)
    slide7.shapes.title.text = "Validation Loss Curves"
    slide7.placeholders[1].text = ""
    
    img_path = "logs/plots/val_loss_curves.png"
    if os.path.exists(img_path):
        slide7.shapes.add_picture(img_path, Inches(1.5), Inches(2.5), width=Inches(7))

    # ---------------------------------------------------------
    # Slide 8: Visual Results - CPU FPS Tradeoffs
    # ---------------------------------------------------------
    slide8 = prs.slides.add_slide(TITLE_SLIDE_LAYOUT)
    slide8.shapes.title.text = "Inference Speed (CPU FPS)"
    slide8.placeholders[1].text = ""
    
    img_path = "logs/plots/cpu_fps_comparison.png"
    if os.path.exists(img_path):
        slide8.shapes.add_picture(img_path, Inches(1.5), Inches(2.5), width=Inches(7))

    # ---------------------------------------------------------
    # Slide 9: Benchmark Results Table
    # ---------------------------------------------------------
    slide9 = prs.slides.add_slide(CONTENT_SLIDE_LAYOUT)
    slide9.shapes.title.text = "Ablation Study Results"
    
    # Add table
    csv_path = "logs/benchmark_results.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        rows, cols = df.shape
        left = Inches(1)
        top = Inches(2.5)
        width = Inches(8)
        height = Inches(1.5)
        
        table = slide9.shapes.add_table(rows + 1, cols, left, top, width, height).table
        
        # Set column names
        for i, col_name in enumerate(df.columns):
            table.cell(0, i).text = col_name
            
        # Set data
        for i in range(rows):
            for j in range(cols):
                val = df.iloc[i, j]
                if isinstance(val, float):
                    table.cell(i + 1, j).text = f"{val:.2f}"
                else:
                    table.cell(i + 1, j).text = str(val)
    else:
        slide9.placeholders[1].text = "Benchmark CSV not found."

    # ---------------------------------------------------------
    # Slide 10: Results & Discussion
    # ---------------------------------------------------------
    slide10 = prs.slides.add_slide(CONTENT_SLIDE_LAYOUT)
    slide10.shapes.title.text = "Results & Trade-off Analysis"
    
    tf = slide10.placeholders[1].text_frame
    tf.text = "Accuracy vs. Computational Overhead"
    
    p = tf.add_paragraph()
    p.text = "Mish & SiLU: Smooth, non-monotonic activations proved better for feature extraction in the shallow 8-layer network, leading to better loss convergence."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "The Mathematical Penalty: Computing 'tanh', 'sigmoid', and 'exp' operations on CPU significantly reduced the FPS compared to the baseline LeakyReLU."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "HardSwish: Emerged as an excellent compromise, providing near-SiLU accuracy while reclaiming FPS lost to exponential math due to its piecewise-linear nature."
    p.level = 1

    # ---------------------------------------------------------
    # Slide 11: Conclusion & Future Work
    # ---------------------------------------------------------
    slide11 = prs.slides.add_slide(CONTENT_SLIDE_LAYOUT)
    slide11.shapes.title.text = "Conclusion & Future Work"
    
    tf = slide11.placeholders[1].text_frame
    tf.text = "Conclusion"
    
    p = tf.add_paragraph()
    p.text = "The PyTorch rewrite successfully modernized YOLO-LITE, enabling rapid ablation studies."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "For strict non-GPU edge deployment, HardSwish or LeakyReLU remain the most viable options to maintain >10 FPS."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Future Work:"
    p.level = 0
    
    p = tf.add_paragraph()
    p.text = "Deploy the newly trained PyTorch weights back to TensorFlow.js (TFJS) to update the web implementation."
    p.level = 1
    
    p = tf.add_paragraph()
    p.text = "Experiment with lightweight Spatial Attention Modules (e.g., CBAM) to increase mAP without adding deep convolutional layers."
    p.level = 1

    # Save presentation
    out_path = "YOLO_LITE_Ablation_Study_Presentation.pptx"
    prs.save(out_path)
    print(f"Presentation generated successfully at: {out_path}")

if __name__ == "__main__":
    create_presentation()
