from time import strftime
import os, sys


from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path


def avatar_stream_generator(
    pic_path,
    audio_path,
    pose_style=1,
    device="cuda",
    size=256,
    still=False,
    unstable=False,
    preprocess="crop",
    expression_scale=1.0,
    batch_size=2,
    enhancer=None,
    background_enhancer=None,
):
    # check if pose style is valid
    if pose_style > 46 or pose_style < 0:
        print("Invalid pose")
        return

    # Get current root and sadtalker paths
    current_root_path = os.path.split(sys.argv[0])[0]

    # get checkpoint and save directory
    checkpoint_dir = "./checkpoints/"
    save_dir = os.path.join("./result", strftime("%Y_%m_%d_%H.%M.%S"))

    sadtalker_paths = init_path(
        checkpoint_dir,
        os.path.join(current_root_path, "src/config"),
        size,
        unstable,
        preprocess,
    )

    # init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths, device)

    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    # crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, "first_frame_dir")
    os.makedirs(first_frame_dir, exist_ok=True)
    print("3DMM Extraction for source image")
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path,
        first_frame_dir,
        preprocess,
        source_image_flag=True,
        pic_size=size,
    )
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    # NOTE: eyeblinking and pose can be adjusted if a reference is wanted, did not migrate, so both are left as none
    ref_eyeblink_coeff_path = None
    ref_pose_coeff_path = None

    # audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path)
    coeff_path = audio_to_coeff.generate(
        batch, save_dir, pose_style, ref_pose_coeff_path
    )

    # 3dface render(not used)
    # if face3dvis:
    #     from src.face3d.visualize import gen_composed_video
    #
    #     gen_composed_video(
    #         args,
    #         device,
    #         first_coeff_path,
    #         coeff_path,
    #         audio_path,
    #         os.path.join(save_dir, "3dface.mp4"),
    #     )

    # coeff2video
    data = get_facerender_data(
        coeff_path,
        crop_pic_path,
        first_coeff_path,
        audio_path,
        batch_size,
        expression_scale=expression_scale,
        still_mode=still,
        preprocess=preprocess,
        size=size,
    )

    for image in animate_from_coeff.generate(
        data,
        save_dir,
        pic_path,
        crop_info,
        enhancer=enhancer,
        background_enhancer=background_enhancer,
        preprocess=preprocess,
        img_size=size,
    ):
        yield image
