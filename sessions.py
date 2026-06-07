import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from datasets.CILdataset import (
    CO3DCIL,
    CO3DCIL_S2C,
    ModelNet40AlignCIL,
    ScanObjNN,
    SessionMaker,
    ShapeNetCIL,
)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_ROOTS = {
    "shapenet": os.path.join(ROOT_DIR, "data", "shapenet"),
    "co3d": os.path.join(ROOT_DIR, "data", "co3d"),
    "modelnet": os.path.join(ROOT_DIR, "data", "modelnet"),
    "scanobjnn": os.path.join(ROOT_DIR, "data", "scanobjnn"),
}

ENV_ROOT_KEYS = {
    "shapenet": "POINT_UQ_SHAPENET_ROOT",
    "co3d": "POINT_UQ_CO3D_ROOT",
    "modelnet": "POINT_UQ_MODELNET_ROOT",
    "scanobjnn": "POINT_UQ_SCANOBJNN_ROOT",
}

DATASET_ALIASES = {
    "shape": "shapenet",
    "shapenet55": "shapenet",
    "co3d_s2c": "co3d",
    "modelnet40": "modelnet",
    "scanobj": "scanobjnn",
    "scanobj-nn": "scanobjnn",
}

SUPPORTED_DATASET_CHOICES = (
    "shapenet",
    "co3d",
    "modelnet",
    "scanobjnn",
    "null",
)


def normalize_dataset_name(name: str) -> str:
    if not name:
        raise ValueError("dataset name cannot be empty")
    normalized = name.strip().lower().replace("-", "_")
    normalized = DATASET_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_DATASET_CHOICES:
        supported = ", ".join(SUPPORTED_DATASET_CHOICES)
        raise ValueError(f"Unsupported dataset name '{name}'. Supported names: {supported}")
    return normalized


@dataclass(frozen=True)
class DatasetRoots:
    shapenet: str
    co3d: str
    modelnet: str
    scanobjnn: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "shapenet": self.shapenet,
            "co3d": self.co3d,
            "modelnet": self.modelnet,
            "scanobjnn": self.scanobjnn,
        }

    def path_for(self, dataset_name: str) -> Optional[str]:
        dataset_name = normalize_dataset_name(dataset_name)
        if dataset_name == "null":
            return None
        return self.as_dict()[dataset_name]


def resolve_roots(
    shapenet_root: Optional[str] = None,
    co3d_root: Optional[str] = None,
    modelnet_root: Optional[str] = None,
    scanobjnn_root: Optional[str] = None,
) -> DatasetRoots:
    return DatasetRoots(
        shapenet=os.environ.get(ENV_ROOT_KEYS["shapenet"], shapenet_root or DEFAULT_ROOTS["shapenet"]),
        co3d=os.environ.get(ENV_ROOT_KEYS["co3d"], co3d_root or DEFAULT_ROOTS["co3d"]),
        modelnet=os.environ.get(ENV_ROOT_KEYS["modelnet"], modelnet_root or DEFAULT_ROOTS["modelnet"]),
        scanobjnn=os.environ.get(ENV_ROOT_KEYS["scanobjnn"], scanobjnn_root or DEFAULT_ROOTS["scanobjnn"]),
    )


def build_roots_from_args(args) -> DatasetRoots:
    return resolve_roots(
        shapenet_root=getattr(args, "shapenet_root", None),
        co3d_root=getattr(args, "co3d_root", None),
        modelnet_root=getattr(args, "modelnet_root", None),
        scanobjnn_root=getattr(args, "scanobjnn_root", None),
    )


def _append_shapenet(session_maker: SessionMaker, roots: DatasetRoots, banlist):
    train = ShapeNetCIL(root=roots.shapenet, partition="train", banlist=banlist)
    test = ShapeNetCIL(root=roots.shapenet, partition="test", banlist=banlist)
    session_maker.append_dataset_train_test(train, test, ShapeNetCIL.id2name)


def _append_modelnet(session_maker: SessionMaker, roots: DatasetRoots, banlist, pt_num: int):
    train = ModelNet40AlignCIL(root=roots.modelnet, partition="train", banlist=banlist, pt_num=pt_num)
    test = ModelNet40AlignCIL(root=roots.modelnet, partition="test", banlist=banlist, pt_num=pt_num)
    session_maker.append_dataset_train_test(train, test, ModelNet40AlignCIL.id2name, train.get_load_method())


def _append_co3d_s2c(session_maker: SessionMaker, roots: DatasetRoots, banlist, pt_num: int):
    train = CO3DCIL_S2C(root=roots.co3d, partition="train", banlist=banlist, pt_num=pt_num)
    test = CO3DCIL_S2C(root=roots.co3d, partition="test", banlist=banlist, pt_num=pt_num)
    session_maker.append_dataset_train_test(train, test, CO3DCIL_S2C.id2name)


def _append_co3d(session_maker: SessionMaker, roots: DatasetRoots, banlist, pt_num: int):
    train = CO3DCIL(root=roots.co3d, partition="train", banlist=banlist, pt_num=pt_num)
    test = CO3DCIL(root=roots.co3d, partition="test", banlist=banlist, pt_num=pt_num)
    session_maker.append_dataset_train_test(train, test, CO3DCIL.id2name)


def _append_scanobjnn(session_maker: SessionMaker, roots: DatasetRoots, banlist, pt_num: int):
    train = ScanObjNN(root=roots.scanobjnn, partition="train", banlist=banlist, pt_num=pt_num)
    test = ScanObjNN(root=roots.scanobjnn, partition="test", banlist=banlist, pt_num=pt_num)
    session_maker.append_dataset_train_test(train, test, ScanObjNN.id2name)


def _reorder_session_categories(session_maker: SessionMaker, ordered_names):
    if len(ordered_names) != session_maker.cat_tot:
        raise ValueError(
            f"Expected {session_maker.cat_tot} category names for reordering, got {len(ordered_names)}"
        )

    if len(set(ordered_names)) != len(ordered_names):
        raise ValueError("Duplicate category names found in requested session reorder")

    missing = [name for name in ordered_names if name not in session_maker.name2id]
    if missing:
        raise ValueError(f"Cannot reorder session; missing categories: {missing}")

    old_indices = [session_maker.name2id[name] for name in ordered_names]
    if len(set(old_indices)) != len(old_indices):
        raise ValueError("Duplicate source indices found while reordering session categories")

    session_maker.id2name = [session_maker.id2name[idx] for idx in old_indices]
    session_maker.data_train = [session_maker.data_train[idx] for idx in old_indices]
    session_maker.data_test = [session_maker.data_test[idx] for idx in old_indices]
    session_maker.cat_cnt_train = [session_maker.cat_cnt_train[idx] for idx in old_indices]
    session_maker.cat_cnt_test = [session_maker.cat_cnt_test[idx] for idx in old_indices]
    session_maker.name2id = {name: idx for idx, name in enumerate(session_maker.id2name)}


def _build_shapenet2co3d(roots: DatasetRoots) -> SessionMaker:
    session_maker = SessionMaker()
    shapenet_banlist = [
        "chair",
        "car",
        "sofa",
        "bench",
        "bottle",
        "laptop",
        "telephone",
        "motorcycle",
        "bowl",
        "microwave",
        "skateboard",
        "bag",
        "earphone",
        "remote control",
        "computer keyboard",
        "cellular telephone",
    ]
    _append_shapenet(session_maker, roots, banlist=shapenet_banlist)
    _append_co3d_s2c(session_maker, roots, banlist=["toytrain"], pt_num=1024)
    session_maker.set_session(num_base_cat=39, num_inc_cat=5)
    return session_maker


def _build_modelnet2null(roots: DatasetRoots) -> SessionMaker:
    session_maker = SessionMaker()
    _append_modelnet(session_maker, roots, banlist=[], pt_num=10000)
    _reorder_session_categories(
        session_maker,
        [
            "chair",
            "sofa",
            "airplane",
            "bookshelf",
            "bed",
            "vase",
            "monitor",
            "table",
            "toilet",
            "bottle",
            "mantel",
            "tv_stand",
            "plant",
            "piano",
            "car",
            "desk",
            "dresser",
            "night_stand",
            "glass_box",
            "guitar",
            "range_hood",
            "bench",
            "cone",
            "tent",
            "flower_pot",
            "laptop",
            "keyboard",
            "curtain",
            "bathtub",
            "sink",
            "lamp",
            "stairs",
            "door",
            "radio",
            "xbox",
            "stool",
            "person",
            "wardrobe",
            "cup",
            "bowl",
        ],
    )
    session_maker.set_session(num_base_cat=20, num_inc_cat=5)
    return session_maker


def _build_shapenet2null(roots: DatasetRoots) -> SessionMaker:
    session_maker = SessionMaker()
    _append_shapenet(session_maker, roots, banlist=[])
    session_maker.set_session(num_base_cat=25, num_inc_cat=5)
    return session_maker


def _build_shapenet2scanobjnn(roots: DatasetRoots) -> SessionMaker:
    session_maker = SessionMaker()
    shapenet_banlist = [
        "bag",
        "cabinet",
        "chair",
        "display",
        "table",
        "bed",
        "pillow",
        "sofa",
        "ashcan",
        "bookshelf",
        "mailbox",
    ]
    _append_shapenet(session_maker, roots, banlist=shapenet_banlist)
    _append_scanobjnn(session_maker, roots, banlist=[], pt_num=2048)
    session_maker.set_session(num_base_cat=44, num_inc_cat=5)
    return session_maker


def _build_modelnet2scanobjnn(roots: DatasetRoots) -> SessionMaker:
    session_maker = SessionMaker()
    modelnet_banlist = [
        "bed",
        "bench",
        "bookshelf",
        "chair",
        "desk",
        "door",
        "dresser",
        "monitor",
        "sink",
        "sofa",
        "stool",
        "table",
        "toilet",
        "wardrobe",
    ]
    scanobjnn_banlist = ["bag", "bin", "box", "pillow"]
    _append_modelnet(session_maker, roots, banlist=modelnet_banlist, pt_num=1024)
    _append_scanobjnn(session_maker, roots, banlist=scanobjnn_banlist, pt_num=1024)
    session_maker.set_session(num_base_cat=26, num_inc_cat=4)
    return session_maker


def _build_co3d2null(roots: DatasetRoots) -> SessionMaker:
    session_maker = SessionMaker()
    _append_co3d(session_maker, roots, banlist=["toytrain"], pt_num=1024)
    session_maker.set_session(num_base_cat=25, num_inc_cat=5)
    return session_maker


SESSION_BUILDERS: Dict[Tuple[str, str], Callable[[DatasetRoots], SessionMaker]] = {
    ("shapenet", "co3d"): _build_shapenet2co3d,
    ("shapenet", "scanobjnn"): _build_shapenet2scanobjnn,
    ("modelnet", "scanobjnn"): _build_modelnet2scanobjnn,
    ("shapenet", "null"): _build_shapenet2null,
    ("modelnet", "null"): _build_modelnet2null,
    ("co3d", "null"): _build_co3d2null,
}


def supported_session_pairs():
    return tuple(f"{source}->{target}" for source, target in SESSION_BUILDERS)


def build_session(base_dataset: str, incremental_dataset: str, roots: Optional[DatasetRoots] = None) -> SessionMaker:
    base_dataset = normalize_dataset_name(base_dataset)
    incremental_dataset = normalize_dataset_name(incremental_dataset)
    builder = SESSION_BUILDERS.get((base_dataset, incremental_dataset))
    if builder is None:
        supported = ", ".join(supported_session_pairs())
        raise ValueError(
            f"Unsupported session pair '{base_dataset}->{incremental_dataset}'. Supported pairs: {supported}"
        )
    return builder(roots or resolve_roots())


def shapenet2co3d(shapenet_root=None, co3d_root=None):
    roots = resolve_roots(shapenet_root=shapenet_root, co3d_root=co3d_root)
    return build_session("shapenet", "co3d", roots)


def modelnet2null(modelnet_root=None):
    roots = resolve_roots(modelnet_root=modelnet_root)
    return build_session("modelnet", "null", roots)


def shapenet2null(shapenet_root=None):
    roots = resolve_roots(shapenet_root=shapenet_root)
    return build_session("shapenet", "null", roots)


def shapenet2scanobjnn(shapenet_root=None, scanobjnn_root=None):
    roots = resolve_roots(shapenet_root=shapenet_root, scanobjnn_root=scanobjnn_root)
    return build_session("shapenet", "scanobjnn", roots)


def modelnet2scanobjnn(modelnet_root=None, scanobjnn_root=None):
    roots = resolve_roots(modelnet_root=modelnet_root, scanobjnn_root=scanobjnn_root)
    return build_session("modelnet", "scanobjnn", roots)


def co3d2null(co3d_root=None):
    roots = resolve_roots(co3d_root=co3d_root)
    return build_session("co3d", "null", roots)
