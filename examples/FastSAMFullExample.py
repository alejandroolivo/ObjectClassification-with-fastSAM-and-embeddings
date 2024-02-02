import torch
from PIL import Image
from FastSAM.fastsam import FastSAM, FastSAMPrompt
from FastSAM.utils.tools import convert_box_xywh_to_xyxy
import ast

def main():
    # Define los argumentos aquí, en lugar de usar argparse
    args = {
        "model_path": "./models/FastSAM.pt",
        "img_path": "./key_frames/frame_v1_1.jpg",
        "imgsz": 256,
        "iou": 0.9,
        "text_prompt": None,  # Puedes cambiar esto según sea necesario
        "conf": 0.4,
        "output": "./output/",
        "randomcolor": True,
        "point_prompt": "[[0,0]]",
        "point_label": "[0]",
        "box_prompt": "[[0,0,0,0]]",
        "better_quality": False,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "retina": True,
        "withContours": False
    }

    # Adaptar para usar los valores directamente
    model = FastSAM(args["model_path"])
    args["point_prompt"] = ast.literal_eval(args["point_prompt"])
    args["box_prompt"] = convert_box_xywh_to_xyxy(ast.literal_eval(args["box_prompt"]))
    args["point_label"] = ast.literal_eval(args["point_label"])
    input_image = Image.open(args["img_path"])
    input_image = input_image.convert("RGB")
    everything_results = model(
        input_image,
        device=args["device"],
        retina_masks=args["retina"],
        imgsz=args["imgsz"],
        conf=args["conf"],
        iou=args["iou"]
    )
    bboxes, points, point_label = None, None, None
    prompt_process = FastSAMPrompt(input_image, everything_results, device=args["device"])

    # Lógica de decisión basada en los argumentos
    if args["box_prompt"][0][2] != 0 and args["box_prompt"][0][3] != 0:
        ann = prompt_process.box_prompt(bboxes=args["box_prompt"])
        bboxes = args["box_prompt"]
    elif args["text_prompt"] is not None:
        ann = prompt_process.text_prompt(text=args["text_prompt"])
    elif args["point_prompt"][0] != [0, 0]:
        ann = prompt_process.point_prompt(points=args["point_prompt"], pointlabel=args["point_label"])
        points = args["point_prompt"]
        point_label = args["point_label"]
    else:
        ann = prompt_process.everything_prompt()

    # Guardar los resultados
    prompt_process.plot(
        annotations=ann,
        output_path=args["output"] + args["img_path"].split("/")[-1],
        bboxes=bboxes,
        points=points,
        point_label=point_label,
        withContours=args["withContours"],
        better_quality=args["better_quality"],
    )

if __name__ == "__main__":
    main()
