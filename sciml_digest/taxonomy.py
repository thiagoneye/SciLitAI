"""Scientific search taxonomy and thematic priorities for SciLitAI."""

from __future__ import annotations

TARGET_CATEGORIES: tuple[str, ...] = (
    "cs.LG",
    "stat.ML",
    "physics.comp-ph",
    "physics.flu-dyn",
    "cond-mat.soft",
    "math.NA",
    "eess.SY",
    "cs.SE",
    "cs.DC",
)

SEARCH_CLUSTERS: dict[str, tuple[str, ...]] = {
    "sciml_pinns": (
        "Scientific Machine Learning",
        "SciML",
        "Physics-Informed",
        "PINN",
        "PINNs",
        "Structure-Preserving",
        "Hamiltonian Neural Networks",
        "Lagrangian Neural Networks",
        "Universal Differential Equations",
        "UDE",
        "Neural ODE",
        "Collocation Methods",
        "Meshfree Neural",
    ),
    "neural_operators_rom_surrogates": (
        "Neural Operator",
        "FNO",
        "Fourier Neural Operator",
        "DeepONet",
        "Wavelet Neural Operator",
        "WNO",
        "GNO",
        "GINO",
        "Surrogate Modeling",
        "Model Order Reduction",
        "ROM",
        "POD",
        "DMD",
        "SVD",
        "Nonlinear ROM",
        "SINDy",
        "Gaussian Process Regression",
        "GPR",
    ),
    "uq_probabilistic_methods": (
        "Uncertainty Quantification",
        "UQ",
        "Bayesian Neural Networks",
        "BNN",
        "Bayesian Optimization",
        "MCMC",
        "Gaussian Process",
        "Conformal Prediction",
        "Sensitivity Analysis",
        "Epistemic Uncertainty",
        "Aleatoric Uncertainty",
    ),
    "cfd_rheology_fem_generative_flow": (
        "Computational Fluid Dynamics",
        "CFD",
        "Turbulence Modeling",
        "RANS",
        "LES",
        "DNS",
        "Flow Reconstruction",
        "Flow Super-Resolution",
        "Aerodynamic Drag",
        "Fluid-Structure Interaction",
        "Heat Transfer",
        "Rheology",
        "Non-Newtonian Fluid",
        "Polymer Processing",
        "Rubber Processing",
        "Vulcanization",
        "Finite Element",
        "FEM",
        "FEA",
        "Structural Dynamics",
        "Mechanical Vibrations",
        "Generative Flow Modeling",
        "Physics-Informed GAN",
        "Diffusion Models",
    ),
    "industry40_uns_predictive_maintenance": (
        "Digital Twin",
        "Predictive Maintenance",
        "Fault Detection",
        "Remaining Useful Life",
        "RUL",
        "Unified Namespace",
        "UNS",
        "Industrial Data Architecture",
        "Industrial Data Lake",
        "OT/IT Convergence",
        "IIoT",
        "Event-Driven Architecture",
        "MQTT",
        "Sparkplug B",
        "OPC UA",
        "Smart Manufacturing",
    ),
}

CLUSTER_WEIGHTS: dict[str, float] = {
    "sciml_pinns": 5.0,
    "neural_operators_rom_surrogates": 4.0,
    "cfd_rheology_fem_generative_flow": 2.0,
    "industry40_uns_predictive_maintenance": 2.0,
    "uq_probabilistic_methods": 1.0,
}

# As consultas semânticas do OpenAlex são propositalmente descritivas e evitam
# acrônimos curtos e ambíguos usados apenas no scoring local.
OPENALEX_SEMANTIC_QUERIES: dict[str, str] = {
    "sciml_pinns": (
        "Scientific machine learning for physical systems, physics-informed neural "
        "networks, structure-preserving learning, neural differential equations, "
        "and machine learning methods for solving differential equations"
    ),
    "neural_operators_rom_surrogates": (
        "Neural operators, Fourier neural operators, DeepONet, reduced-order "
        "modeling, surrogate models, and data-driven emulators for scientific "
        "and engineering simulations"
    ),
    "uq_probabilistic_methods": (
        "Uncertainty quantification for scientific machine learning, Bayesian "
        "surrogate models, Gaussian process regression, conformal prediction, "
        "and epistemic or aleatoric uncertainty in computational science"
    ),
    "cfd_rheology_fem_generative_flow": (
        "Machine learning for computational fluid dynamics, turbulence, flow "
        "reconstruction, fluid-structure interaction, rheology, finite element "
        "simulation, structural dynamics, and engineering physics"
    ),
    "industry40_uns_predictive_maintenance": (
        "Digital twins, predictive maintenance, fault detection, remaining useful "
        "life, industrial AI, IIoT, unified namespace, event-driven industrial "
        "data architectures, and smart manufacturing"
    ),
}

# Pelo menos uma expressão temática forte deve aparecer em título ou resumo para
# que um trabalho do OpenAlex possa competir no ranking por citações.
OPENALEX_STRONG_TERMS: tuple[str, ...] = (
    "Scientific Machine Learning",
    "Physics-Informed",
    "Hamiltonian Neural",
    "Lagrangian Neural",
    "Universal Differential Equation",
    "Neural ODE",
    "Neural Operator",
    "Fourier Neural Operator",
    "DeepONet",
    "Wavelet Neural Operator",
    "Surrogate Modeling",
    "Model Order Reduction",
    "Proper Orthogonal Decomposition",
    "Dynamic Mode Decomposition",
    "Gaussian Process Regression",
    "Uncertainty Quantification",
    "Bayesian Neural Network",
    "Conformal Prediction",
    "Epistemic Uncertainty",
    "Aleatoric Uncertainty",
    "Computational Fluid Dynamics",
    "Turbulence Modeling",
    "Flow Reconstruction",
    "Flow Super-Resolution",
    "Aerodynamic Drag",
    "Fluid-Structure Interaction",
    "Non-Newtonian Fluid",
    "Finite Element",
    "Structural Dynamics",
    "Mechanical Vibrations",
    "Digital Twin",
    "Predictive Maintenance",
    "Fault Detection",
    "Remaining Useful Life",
    "Unified Namespace",
    "Industrial Data Architecture",
    "Industrial Data Lake",
    "OT/IT Convergence",
    "Smart Manufacturing",
)

OPENALEX_ALLOWED_WORK_TYPES: frozenset[str] = frozenset({"article", "preprint"})
