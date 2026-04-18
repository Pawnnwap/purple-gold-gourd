import warnings

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 .* doesn't match a supported version!",
)

__all__ = ["__version__"]

__version__ = "0.1.2"
__title__ = "紫金葫芦"
__distribution_name__ = "purple-gold-gourd"
