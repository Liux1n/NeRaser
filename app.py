import time
import zipfile
import logging
from io import BytesIO
import tempfile
from functools import lru_cache
from pathlib import Path
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import streamlit.components.v1 as components
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from enum import Enum

class AppSteps(str):
    S0 = "0. Intro 👐"
    S1 = "1. Upload Data 🎥"
    S2 = "2. Generate Object Mask 🎭"
    S3 = "3. Inspect Object Mask for Each Frame 👓"
    S4 = "4. Train NeRF on Scene 🏃‍♀️"
    S5 = "5. View Training Result :100:"


def setup_logging():
    our_logger = logging.getLogger("neraser")
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s:%(name)s:%(funcName)s() %(message)s',
    )
    # only show our log
    for v in logging.Logger.manager.loggerDict.values():
        if isinstance(v, logging.Logger) and v.name != our_logger.name:
            v.disabled = True
    our_logger.setLevel(logging.DEBUG)
    return our_logger


logger = setup_logging()


def cached_imread(p: Path):
    @st.cache_data
    def impl(p):
        ret = np.asarray(Image.open(p))
        return ret
    return impl(p)


def frame_id_png(frame_id: int, ext:str="png") -> str:
    return f"frame_{frame_id:05d}.{ext}"


@st.cache_resource # should only be called once
def get_tempdir():
    def impl():
        dir = tempfile.TemporaryDirectory(
            suffix="_neraser",
        )
        logger.warning(f"creating tempdir {dir.name}")
        return dir
    return impl()


def zip_upload():
    st.header(
        AppSteps.S1,
        anchor="capture",
    )
    st.write("""
        Please use [Polycam](https://apps.apple.com/us/app/polycam-3d-scanner-lidar-360/id1532482376)
        to capture a 360° view of the object that you wish to erase. Ideally the object should rest on
        a flat surface. Try to pan the camera around the object at a slow and steady pace, and capture
        a complete view of the object, to achieve better erasure result. Be careful to no obstruct
        any light source and avoid casting large shadow onto the scene. Upload the zip file produced by
        Polycam here.
    """)
    @st.cache_resource
    def validate_zipfile(input_file):
        try:
            with zipfile.ZipFile(file=input_file) as f:
                output_dir = get_tempdir().name
                dataset_name = set([x.split('/')[0] for x in f.namelist() if x.endswith('/')])
                if len(dataset_name) != 1:
                    raise ValueError("zipfile contains more than 1 dataset")
                dataset_name = str(list(dataset_name)[0])
                f.extractall(output_dir)
            dataset_dir = Path(output_dir) / dataset_name
            return dataset_dir
        except zipfile.BadZipFile:
            st.error("Invalid zipfile, please check again")


    input_file = st.file_uploader(
        "Upload Polycam ZIP file :compression:",
        type="zip",
        help="upload a ZIP file no more than 1024MB",
        accept_multiple_files=False,
    )
    def validate_dataset():
        dataset_name = st.session_state.dataset_dir.stem
        images_dir = st.session_state.dataset_dir / "images"
        num_images = sum(1 for _ in images_dir.iterdir())
        st.session_state.num_images = num_images
        st.success(f"Nice job, you have uploaded data for `{dataset_name}`; it contains {num_images} keyframes.\n You may proceed to the next step.")

    if input_file is not None or st.button("Or use an example dataset"):
        b = BytesIO(input_file.getvalue()) if input_file is not None else open(Path.home()/"Downloads/polycam_mate_floor2.zip", 'rb')
        dataset_dir = validate_zipfile(b)
        st.session_state.dataset_dir = dataset_dir
        validate_dataset()


def wrap_click(key):
    msg = st.empty()
    msg.write("Use the mouse cursor to select the object to be erased on the image below:")
    example_img = cached_imread(Path(get_tempdir().name)/"polycam_mate_floor2/images"/frame_id_png(key))
    value = streamlit_image_coordinates(
        example_img,
        width=example_img.shape[1],
        height=example_img.shape[0],
    )
    logger.debug(value)
    if value is not None:
        x, y = value["x"], value["y"]
        with msg:
            with st.spinner(f"you have chosen {x=}, {y=}; now DSTT will propagate the object mask to all other frames"):
                time.sleep(2)
            st.success("DSTT propagated masks to all frames, proceed to the next step")
    st.session_state.click_val = value
    return value


def get_object_mask():
    st.header(
        AppSteps.S2,
        anchor="dstt",
    )
    st.write("""
            Using [Decoupled Spatial-Temporal Transformer](https://github.com/ruiliu-ai/DSTT) we generate masks for the object to be erased using just one frame of
            the sequence. Selecte the object in this frame, the algorithm will generate the mask
            and propagate the mask to the rest of the frames.
    """)
    wrap_click(1)


def propagete_object_mask():
    st.header(
        AppSteps.S3,
    )
    st.write("""
        We can now visualze the result of object mask propagation. If everything looks fine,
        proceed to the next step. Otherwise, go to the previous step and regenerate mask.
    """)
    dataset_dir = st.session_state.dataset_dir
    num_images = st.session_state.num_images
    images_dir = dataset_dir / "images"
    def highlight_object(frame_id: int) -> NDArray:
        ret = cached_imread(images_dir/frame_id_png(frame_id)).copy()
        mask = cached_imread(dataset_dir/"masks_2"/frame_id_png(frame_id))
        mask = np.bitwise_not(mask.astype(bool))
        np.putmask(ret[..., 0], mask, 200) # highlight obj in red
        return ret

    frame_id = st.slider(
        label="move the slider to check object mask for each frame #",
        min_value=1,
        max_value=num_images,
        value=1, step=1,
        #on_change=highlight_object,
    )
    obj_img = highlight_object(frame_id)
    st.image(obj_img)


def train1():
    st.header(
        AppSteps.S4,
        anchor="train1",
    )
    st.write("""
            Using [NerfStudio](https://docs.nerf.studio/) we train a NeRF representation of the scene;
            Brew yourself a pot of tea :teapot: as the training takes a while;
    """)
    with st.spinner(f"training NeRF... "):
        time.sleep(5)
    with st.spinner(f"Inpainting area under object, wapring perspectives... "):
        time.sleep(5)
    with st.spinner(f"training NeRF on inpainted result... "):
        time.sleep(5)
    st.success("Done training :tada: we can now visualize the new NeRF scene without the object")
    #vf = open(dataset_dir/"polycam_mate_floor2.mp4", 'rb')
    view()
    st.video("https://youtu.be/AaSgzrqt-gI")

    
st.set_page_config(
    page_title="NeRasor",
    page_icon=":tophat:",
)

def intro():
    st.header(AppSteps.S0)
    st.write("""
        Thank you for trying NeRaser, our app to erase objects from [NeRF](https://en.wikipedia.org/wiki/Neural_radiance_field)
        representation of a scene. To try the app with your own data, you would need to have an iPhone 12 or above with LiDAR;
        or you can proceed with an example dataset that is already prepared, or use a scene from an existing NeRF dataset.
    """)
    st.subheader("Ready to start?")
    st.write("Use the dropdown menu on the left to navigate between steps")
    pass

def view():
    v = "https://viewer.nerf.studio/versions/23-05-15-1/?websocket_url=ws://localhost:7007"
    viewer = components.iframe(
        src=v,
        height=800,
        scrolling=True,
    )

step_name_to_funcs = {
    AppSteps.S0: intro,
    AppSteps.S1: zip_upload,
    AppSteps.S2: get_object_mask,
    AppSteps.S3: propagete_object_mask,
    AppSteps.S4: train1,
    AppSteps.S5: view,
}

step_name = st.sidebar.selectbox(
    "Steps to erase object from NeRF",
    step_name_to_funcs.keys(),
)

step_name_to_funcs[step_name]()
