from traffic import tqdm_style

# Choose the appropriate tqdm based on the style
if tqdm_style == "notebook":
    from tqdm.notebook import tqdm
elif tqdm_style == "rich":
    from tqdm.rich import tqdm
elif tqdm_style == "auto":
    from tqdm.auto import tqdm
elif tqdm_style == "silence":

    def tqdm(iterable, *args, **kwargs):
        return iterable  # Dummy tqdm function
else:
    from tqdm import tqdm  # noqa: F401
