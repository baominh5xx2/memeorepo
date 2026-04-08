"""Dataset abstractions for DAMP-ES."""

__all__ = ["CrossDomainSegDataset"]


def __getattr__(name: str):
	if name == "CrossDomainSegDataset":
		from .crossdomain_seg import CrossDomainSegDataset

		return CrossDomainSegDataset
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
