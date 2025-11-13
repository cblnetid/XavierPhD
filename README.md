# **CLIR: Covariance Lipschitz-Indirect Restriction**

## Official implementation of CLIR (Covariance Lipschitz-Indirect Restriction), a novel method for enhancing neural network robustness against adversarial attacks by indirectly reducing the local Lipschitz constant through variance minimization.

### Paper: "Lipschitz-indirect method for robustness and defense of neural networks"
Authors: Xavier Sierra-Canto, Carlos Brito-Loeza, Ricardo Legarda-Saenz
Journal: Submitted to Neural Computing and Applications

#### 📖 Abstract
Deep neural networks are vulnerable to adversarial examples—carefully crafted inputs that cause misclassification. CLIR addresses this by making networks less sensitive to small input variations through indirect Lipschitz regularization. Instead of explicitly computing the Lipschitz constant (an NP problem), CLIR minimizes the intra-class variance at the network output, which theoretically bounds the local Lipschitz constant and empirically enhances robustness against various adversarial attacks.

![CLIR Animation](media/spiral_no_clir.gif)

#### ✨ Key Features
* Indirect Lipschitz Regularization: Avoids explicit Lipschitz computation by minimizing output variance
* Architecture Agnostic: Works with any neural network architecture (ResNet, EfficientNet, MobileNet, etc.)
* Simple Implementation: Single penalty term added to the loss function
* Comprehensive Defense: Evaluated against 6 adversarial attack methods (FGSM, PGD, CW, DeepFool, JSMA, Gaussian noise)
* Theoretically Grounded: Supported by mathematical proofs linking variance reduction to Lipschitz constant bounds

#### 🛡️ Adversarial Robustness Results
CLIR demonstrates significant improvements in robustness across multiple datasets and architectures:

MNIST Results (Accuracy under attack)
Attack Method	Standard	CLIR	Improvement
FGSM	81%	88%	+7%
PGD	21%	58%	+37%
CW	57%	85%	+28%
*See paper for complete results on CIFAR-10 and ImageNet subsets*

#### 📚 Theoretical Foundation
CLIR is based on three key propositions:

Proposition 1: Bounding the covariance matrix diagonal bounds the distance to class mean
Proposition 2: Bounded distance to mean implies bounded distance between class points
Proposition 3: Bounded inter-point distance reduces the local Lipschitz constant

The method minimizes:

text
Loss_total = Loss_classification + λ * Σ_c ||diag(Σ_c)||_1
where Σ_c is the covariance matrix of class c outputs.

#### 📊 Citation
If you use CLIR in your research, please cite our paper:

bibtex
@article{sierra2024clir,
  title={Lipschitz-indirect method for robustness and defense of neural networks},
  author={Sierra-Canto, Xavier and Brito-Loeza, Carlos and Legarda-Saenz, Ricardo},
  journal={Neural Computing and Applications},
  year={2024},
  publisher={Springer}
}
#### 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

#### 📧 Contact
For questions about this implementation, please open an issue or contact:

Xavier Sierra-Canto: xavier.sierra@utmetropolitana.edu.mx

Carlos Brito-Loeza: carlos.brito@correo.uady.mx

