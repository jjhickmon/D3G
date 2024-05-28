import clip
import torch
import torchvision
import json
from util.evals import *
from util.get_embeddings import get_embeddings, generate_images
from util.equigen_templates import profession_templates, race_7_templates, race_4_templates, gender_templates, age_templates
from util.custom_imagefolder import EquigenImageFolder
from tqdm import tqdm

device = "cuda"
model, preprocess = clip.load("./ViT-L-14.pt")
model.eval()

with open('idenprof/idenprof_model_class.json', 'r') as f:
    prof_class_dict = json.load(f)
with open('idenprof/race_7_class.json', 'r') as f:
    race_7_class_dict = json.load(f)
with open('idenprof/race_4_class.json', 'r') as f:
    race_4_class_dict = json.load(f)
with open('idenprof/gender_class.json', 'r') as f:
    gender_class_dict = json.load(f)
with open('idenprof/age_class.json', 'r') as f:
    age_class_dict = json.load(f)
class_dicts = (race_7_class_dict, race_4_class_dict, gender_class_dict, age_class_dict)

GEN_EMBEDDINGS = True
GEN_IMAGES = False
dataset_root = "idenprof/test/"
gen_img_root = "equigen_images/sdxl_classify_race7_prof_5img/"
fairface_output = "FairFace/idenprof_paths_output.csv"
true_dataset = EquigenImageFolder(root=dataset_root, transform=preprocess, csv_file=fairface_output, class_dicts=class_dicts)

# NOTE: make sure this aligns with true_embeddings
# dataset_classnames = [prof_class_dict[key] for key in sorted(prof_class_dict.keys())]
dataset_classnames = [race_7_class_dict[key] for key in sorted(race_7_class_dict.keys())]
# dataset_classnames = [race_4_class_dict[key] for key in sorted(race_4_class_dict.keys())]
# dataset_classnames = [gender_class_dict[key] for key in sorted(gender_class_dict.keys())]
# dataset_classnames = [age_class_dict[key] for key in sorted(age_class_dict.keys())]

dataset_classnames_aug = [classname + " person" for classname in dataset_classnames]
# dataset_classnames_aug = dataset_classnames

gen_img_prompts = {classname: [template.format(classname) for template in profession_templates] for classname in dataset_classnames}
print(gen_img_prompts)

true_loader = torch.utils.data.DataLoader(true_dataset, batch_size=200, num_workers=8)

with torch.no_grad():
    if GEN_EMBEDDINGS:
        if GEN_IMAGES:
            print("Generating Images")
            generate_images(gen_img_prompts, dataset_classnames, num_gen_samples=5, eval_dir=gen_img_root, device=device)

        gen_dataset = torchvision.datasets.ImageFolder(root=gen_img_root, transform=preprocess)
        gen_loader = torch.utils.data.DataLoader(gen_dataset, batch_size=10*5, num_workers=8)

        print("Generating Embeddings")
        templates = (profession_templates, race_7_templates, race_4_templates, gender_templates, age_templates)
        true_embeddings, gen_text_embeddings, gen_img_embeddings = get_embeddings(true_loader, gen_loader, templates, dataset_classnames, dataset_classnames_aug, model, device)
        true_embeddings_list = [e.tolist() for e in true_embeddings]
        gen_text_embeddings_list = [e.tolist() for e in gen_text_embeddings]
        gen_img_embeddings_list = [e.tolist() for e in gen_img_embeddings]
        with open("util/embeddings/true_embeddings.json", 'w') as f:
            json.dump(true_embeddings_list, f, indent=4)
        with open("util/embeddings/gen_text_embeddings.json", 'w') as f:
            json.dump(gen_text_embeddings_list, f, indent=4)
        with open("util/embeddings/gen_img_embeddings.json", 'w') as f:
            json.dump(gen_img_embeddings_list, f, indent=4)
    else:
        print("Loading Embeddings")
        with open("util/embeddings/true_embeddings.json", 'r') as f:
            t_embeddings = json.load(f)
            true_embeddings = tuple([torch.tensor(embedding).to(device) for embedding in t_embeddings])
        with open("util/embeddings/gen_text_embeddings.json", 'r') as f:
            gt_embeddings = json.load(f)
            gen_text_embeddings = tuple([torch.tensor(embedding).to(device) for embedding in gt_embeddings])
        with open("util/embeddings/gen_img_embeddings.json", 'r') as f:
            gi_embeddings = json.load(f)
            gen_img_embeddings = tuple([torch.tensor(embedding).to(device) for embedding in gi_embeddings])

    print("Getting Results")
    (true_img_embeddings, idenprof_profession_labels, fairface_7_race_labels, fairface_4_race_labels, fairface_gender_labels, fairface_age_labels) = true_embeddings
    (std_text_embeddings, avg_profession_embeddings, avg_race_7_embeddings, avg_race_4_embeddings, avg_gender_embeddings, avg_age_embeddings) = gen_text_embeddings
    (avg_gen_img_embeddings) = gen_img_embeddings

    # NOTE: make sure this aligns with dataset_classnames
    true_embeddings = (true_img_embeddings, fairface_7_race_labels)
    top1, avg_per_class, per_class = standard_clip_baseline(true_embeddings, avg_age_embeddings, dataset_classnames, 1024, device)
    print("Standard CLIP Accuracies: ", "top1=", top1, "avg per-class=", avg_per_class, "per-class=", per_class)

    weights = torch.arange(0, 1.001, .01).tolist()
    all_accs = []
    for weight in tqdm(weights):
        text_weight = weight
        img_weight = 1 - weight
        top1, avg_per_class, per_class = equigen_weighted_sum_embeddings(true_embeddings, avg_age_embeddings, avg_gen_img_embeddings, text_weight, img_weight, dataset_classnames, 1024, device)
        all_accs.append((weight, top1, avg_per_class, per_class))
    best_top_1 = max(all_accs, key=lambda x: x[1])
    print("Equigen Weighted Sum Accuracies: ", "best_weight=", f"(text={best_top_1[0]}, img={1-best_top_1[0]})", "top1=", best_top_1[1], "avg per-class=", best_top_1[2], "per-class=", best_top_1[3])