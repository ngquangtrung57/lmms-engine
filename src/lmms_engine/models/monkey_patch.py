# Most of the code copied from https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/monkey_patch.py
# Modified to work on patch our models

import collections
import contextlib
import inspect

from loguru import logger
from transformers import PreTrainedModel


class MonkeyPatcher:
    def __init__(self, *args, **kwargs):
        # In format
        # {"model_type": {"liger": apply_liger_kernel_to_xxx, "custom": apply_custom_kernel_to_xxx}}
        self._dict = collections.defaultdict(dict)
        # Original (pre-patch) class attributes keyed by (cls, attr_name).
        # Patch functions call stash_original() right before overwriting a
        # class-level method that is incompatible with generation (e.g. the
        # rmpad forwards, which expect packed/unpadded inputs and crash in
        # the kv-cache decode path). First write wins, so re-applying a
        # patch never records an already-patched callable as "original".
        self._stash = {}

    def stash_original(self, cls, attr: str):
        key = (cls, attr)
        if key not in self._stash:
            self._stash[key] = getattr(cls, attr)

    @property
    def has_stash(self) -> bool:
        return bool(self._stash)

    @contextlib.contextmanager
    def generation_context(self):
        """Temporarily restore the stashed original class methods so
        ``model.generate()`` runs the plain HF padded/kv-cache path, then put
        the patched (training) callables back on exit.

        Restores the saved patched callables directly instead of re-invoking
        the apply functions: re-applying would re-run instance-level patching
        and double-wrap forwards (e.g. ``wrap_vit_forward``).
        """
        patched = {}
        for (cls, attr), original in self._stash.items():
            patched[(cls, attr)] = getattr(cls, attr)
            setattr(cls, attr, original)
        try:
            yield
        finally:
            for (cls, attr), fn in patched.items():
                setattr(cls, attr, fn)

    def register(self, model_type, patch_type):
        def decorator(func):
            if not callable(func):
                raise TypeError(f"Error: {func} must be callable!")
            if patch_type in self._dict[model_type]:
                logger.warning(
                    f"Monkey patch for model_type='{model_type}', patch_type='{patch_type}' already exists and will be overwritten by {getattr(func, '__name__', repr(func))}."
                )
            self._dict[model_type][patch_type] = func
            return func

        return decorator

    def apply_monkey_patch(self, model_type, patch_type, **kwargs):
        if isinstance(patch_type, list):
            for patch in patch_type:
                self.apply_monkey_patch(model_type, patch, **kwargs)
            return
        if not model_type:
            logger.info("Model type was not provided. No patches will be applied.")
            return
        if model_type not in self._dict.keys():
            logger.info(
                f"There are currently no patches supported for model type: {model_type} with patch type: {patch_type}. Available model types: {self._dict.keys()}"
            )
            return
        if patch_type not in self._dict[model_type]:
            logger.info(
                f"Patch type {patch_type!r} not registered for model type {model_type!r}; skipping. "
                f"Available patch types: {list(self._dict[model_type].keys())}"
            )
            return

        apply_fn = self._dict[model_type][patch_type]
        apply_fn_signature = inspect.signature(apply_fn)

        # Filter out the keyword arguments that are not supported by the apply function
        applicable_kwargs = {key: value for key, value in kwargs.items() if key in apply_fn_signature.parameters}

        logger.info(
            f"Applying patches for model type: {model_type} with patch type: {patch_type} with kwargs: {applicable_kwargs}"
        )

        apply_fn(**applicable_kwargs)

    def apply_monkey_patch_to_instance(self, model: PreTrainedModel, patch_type, **kwargs):
        if isinstance(patch_type, list):
            for patch in patch_type:
                self.apply_monkey_patch_to_instance(model, patch, **kwargs)
            return

        model_type = getattr(model, "config", None) and getattr(model.config, "model_type", None)
        if not model_type:
            logger.info("Model type could not be determined from model config. No patches will be applied.")
            return
        if model_type not in self._dict.keys():
            logger.info(
                f"There are currently no patches supported for model type: {model_type} with patch type: {patch_type}. Available model types: {self._dict.keys()}"
            )
            return
        if patch_type not in self._dict[model_type]:
            logger.info(
                f"Patch type {patch_type!r} not registered for model type {model_type!r}; skipping. "
                f"Available patch types: {list(self._dict[model_type].keys())}"
            )
            return

        apply_fn = self._dict[model_type][patch_type]

        apply_fn_signature = inspect.signature(apply_fn)

        # Filter out the keyword arguments that are not supported by the apply function
        applicable_kwargs = {key: value for key, value in kwargs.items() if key in apply_fn_signature.parameters}
        logger.info(
            f"Applying patches to model instance with model type: {model_type} with patch type: {patch_type} with kwargs: {applicable_kwargs}"
        )

        apply_fn(model=model, **applicable_kwargs)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def __str__(self):
        return str(self._dict)

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    def items(self):
        return self._dict.items()


MONKEY_PATCHER = MonkeyPatcher()
