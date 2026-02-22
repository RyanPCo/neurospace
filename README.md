
## Inspiration

Brain tumor diagnosis from MRI is a long and challenging process, and black box machine learning algorithms are hard to trust in such high-stakes clinical settings. We wanted to build something that does more than output a class label: a system that helps doctors inspect, challenge, and improve the model into something they can trust while also dramatically speeding up review by processing hundreds of MRI slices in seconds.

Neurospace was inspired by a simple idea: if clinicians can see what the model is looking at, annotate what matters, and immediately retrain/validate in one workflow, AI becomes a collaborative tool instead of a hidden decision-maker. The result is explainable AI that supports clinical reasoning for 3D brain tumor diagnosis, enabling doctors to fully control how the model operates while reducing time spent on manual slice-by-slice review.

## What it does

Neuroscope is an end-to-end explainable AI platform for brain tumor MRI analysis. It combines validation, annotation, retraining, interpretability tools, and vector similarity search for case-based comparison. It is designed for rapid review, processing hundreds of MRI images/slices in seconds to support faster clinical decision-making.

- **Validation browser:** Browse indexed brain tumor MRI slices, see model predictions and confidence, and quickly inspect errors.
- **Similarity search:** Embed MRI scans/slices with DINOv3 and retrieve visually similar cases from Actian VectorAI DB to provide clinical context during validation.
- **Annotation tools:** Draw polygons/brush annotations directly on images; annotations feed into training through an annotation-weighted loss.
- **Grad-CAM overlays:** Visualize activation maps on slices to see where the model attends.
- **3D Grad-CAM pipeline:** Generate full 3D heatmaps and overlays from a trained 2D CNN (slice-by-slice Grad-CAM aggregation), effectively creating a 3D segmentation-like view from 2D CNN outputs.

## How we built it
We used a variety of machine learning techniques to build Neurospace. For visualization, we first built a custom grayscale 2D CNN and applied GradCAM to it, allowing doctors to "see" what part of a scan was most influential for the model's decision. We use these scans from three angles (axial/coronal/sagittal) to construct a 3D render of the brain, using splining to fill in minor inconsistencies, creating a full 3D heatmap for the user without running an expensive and impractical 3D CNN.

The primary backend innovation was our annotated-weighted loss, which augments standard classification with clinical spatial supervision. Using an algorithm developed in the Harvard paper "Right for the Right Reasons," we create masks of regions drawn on scans by doctors, resize them to feature-map resolution, and use backpropagation to update the weights in a way that optimally adjusts the model to fit the doctor's edits.

To add case-based context, we use DINOv3 embeddings stored in an Actian VectorAI DB for MRI similarity search. During validation, the system retrieves visually similar scans so users can compare predictions against analogous cases, which improves error analysis and trust.

We used FastAPI, SQLite, and WebSockets for backend/routing and React/TypeScript for the frontend.

## Challenges we ran into

- **Turning free-form annotations into stable training supervision:** Converting polygons/brush strokes into masks, then aligning them with intermediate feature maps at different resolutions, required careful rasterization and interpolation.
- **Balancing classification and spatial supervision:** The annotation-weighted loss is powerful, but tuning the spatial term was nontrivial. Too much weight hurts classification; too little has minimal effect. We had to revisit the differential equations backing CNNs and build a Lagrangian-based optimizer model on top to guide retraining.
- **Making 3D Grad-CAM actually readable:** Slice-wise Grad-CAM can become noisy when aggregated into 3D. We had to add brain masking and percentile filtering (per-slice and global) to reduce artifacts and make the output useful in practice.
- **Retrieval quality for MRI similarity search:** Integrating DINOv3 embeddings with Actian VectorAI DB worked well, but getting clinically meaningful similarity (not just visually similar intensity patterns) is an ongoing tuning challenge.

## Accomplishments that we're proud of

- **Keeping human doctors in the loop:** Neuroscope is more than a classifier demo: users can validate predictions, retrieve similar cases, annotate errors, retrain with annotation-weighted loss, and immediately re-validate.
- **Clinician-guided attention shaping:** Our annotation-weighted loss lets doctors directly influence model attention—teaching the model where to focus, not only what label to output.
- **3D interpretability from a 2D CNN:** We implemented a practical 3D Grad-CAM pipeline (with masking + percentile cleanup) and integrated it into a viewer, without requiring a heavy 3D network.
- **Fast AI-assisted review at scale:** Neuroscope can process hundreds of MRI slices in seconds, enabling much faster first-pass review than manual slice-by-slice inspection while keeping the clinician in control.

## What we learned

- **Trust is key in the hospital:** Clinicians need tools to inspect, compare, and correct model behavior.
- **Spatial supervision is valuable clinical signal:** Doctor annotations encode intent (“look here,” “not there”), and incorporating that into the loss can better align model attention with human reasoning.
- **Retrieval adds important interpretability context:** Heatmaps explain where the model is focusing, but similarity search explains what this case resembles. Combining DINOv3 with Actian VectorAI DB made validation and error analysis much more useful.
- **Raw explainability outputs need cleanup:** Grad-CAM is helpful, but post-processing (brain masking, thresholding, normalization) is essential to make outputs readable and actionable.
- **Product UX matters as much as the ML:** Fast annotation, clear visualization, smooth training controls, and near-instant predictions are critical if explainable AI is going to be used in real workflows.

## What's next for Neuroscope

- **Improve retrieval quality:** Expand Actian VectorAI DB + DINOv3 to full-volume/case-level retrieval, hybrid metadata + vector search, and clinician-feedback-guided ranking.
- **Stronger models:** Add segmentation and explore native 3D models while keeping our current 2D + 3D Grad-CAM pipeline as a lightweight baseline.
- **Smarter annotation loop:** Add active learning, multi-rater support, and annotation versioning/auditing.
- **Better explainability:** Add more attribution methods and quantitatively evaluate explanations against expert annotations.
- **Clinical validation:** Run clinician usability studies and benchmark on more datasets / real-world cases.
