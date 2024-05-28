import torch
import clip
import os
from stable_diffusion.stable_diffusion_xl import init_base, init_refiner, sdxl_generate_image
from tqdm import tqdm

def generate_images(prompts, dataset_classnames, num_gen_samples=1, eval_dir=None, device="cuda"):
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    sdxl_base = init_base("stable_diffusion/stable-diffusion-xl-base-1.0", device=device)
    sdxl_refiner = init_refiner("stable_diffusion/stable-diffusion-xl-refiner-1.0", sdxl_base, device=device)
    sdxl_base.set_progress_bar_config(disable=True)
    sdxl_refiner.set_progress_bar_config(disable=True)

    for classname in tqdm(list(prompts.keys())[2:], desc="Generating images"):
        if not os.path.exists(os.path.join(eval_dir, classname)):
            os.makedirs(os.path.join(eval_dir, classname))

        for prompt in prompts[classname]:
            print(prompt.strip())
            gen_imgs = sdxl_generate_image(prompt.strip(), None, base=sdxl_base, refiner=sdxl_refiner, n_steps=50, guidance_scale=15, num_images_per_prompt=num_gen_samples, seed=0, device=device)
            for i, gen_img in enumerate(gen_imgs):
                filename = prompt.replace('A photo of a ', '').replace('A photo of an ', '').replace('/', '_').replace(' ', '-')
                save_path = os.path.join(eval_dir, classname, f"{filename}-{i}.jpg")
                gen_img.save(save_path)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()


def get_embeddings(true_loader, gen_loader, templates, dataset_classnames, dataset_classnames_aug, model, device):
    (profession_templates, race_7_templates, race_4_templates, gender_templates, age_templates) = templates

    true_img_embeddings, idenprof_profession_labels, fairface_7_race_labels, fairface_4_race_labels, fairface_gender_labels, fairface_age_labels = [], [], [], [], [], []
    for true_images, idenprof_profession, fairface_7_race, fairface_4_race, fairface_gender, fairface_age in true_loader:
        true_image_embeddings = model.encode_image(true_images.to(device))
        true_image_embeddings /= true_image_embeddings.norm(dim=1, keepdim=True)
        true_img_embeddings.append(true_image_embeddings)
        idenprof_profession_labels.append(idenprof_profession)
        fairface_7_race_labels.append(fairface_7_race)
        fairface_4_race_labels.append(fairface_4_race)
        fairface_gender_labels.append(fairface_gender)
        fairface_age_labels.append(fairface_age)

    avg_gen_img_embeddings = []
    for gen_images, gen_labels in gen_loader:
        gen_img_embedding = model.encode_image(gen_images.to(device))
        gen_img_embedding /= gen_img_embedding.norm(dim=-1, keepdim=True)
        avg_gen_img_embedding = gen_img_embedding.mean(dim=0, keepdim=True)
        avg_gen_img_embedding /= avg_gen_img_embedding.norm(dim=1, keepdim=True)
        avg_gen_img_embeddings.append(avg_gen_img_embedding)

    std_text_embeddings, avg_profession_embeddings, avg_race_7_embeddings, avg_race_4_embeddings, avg_gender_embeddings, avg_age_embeddings = [], [], [], [], [], []
    for label in range(len(dataset_classnames)):
        if label == 0:
            print(f"------{dataset_classnames[label]}------")
            print("std", f"A photo of a {dataset_classnames_aug[label]}")
        classnames = [f"A photo of a {dataset_classnames_aug[label]}"]
        std_text_embedding = model.encode_text(clip.tokenize(classnames).to(device))
        std_text_embedding /= std_text_embedding.norm(dim=1, keepdim=True)
        std_text_embeddings.append(std_text_embedding)

        template_texts = [template.format(dataset_classnames[label]) for template in profession_templates]
        if label == 0:
            print("prof", template_texts[0])
        template_texts = clip.tokenize(template_texts).to(device)
        class_text_embedding = model.encode_text(template_texts)
        class_text_embedding = class_text_embedding.mean(dim=0, keepdim=True)
        class_text_embedding /= class_text_embedding.norm(dim=1, keepdim=True)
        avg_profession_embeddings.append(class_text_embedding)

        template_texts = [template.format(dataset_classnames[label]) for template in race_7_templates]
        if label == 0:
            print("race7", template_texts[0])
        template_texts = clip.tokenize(template_texts).to(device)
        class_text_embedding = model.encode_text(template_texts)
        class_text_embedding = class_text_embedding.mean(dim=0, keepdim=True)
        class_text_embedding /= class_text_embedding.norm(dim=1, keepdim=True)
        avg_race_7_embeddings.append(class_text_embedding)

        template_texts = [template.format(dataset_classnames[label]) for template in race_4_templates]
        if label == 0:
            print("race4", template_texts[0])
        template_texts = clip.tokenize(template_texts).to(device)
        class_text_embedding = model.encode_text(template_texts)
        class_text_embedding = class_text_embedding.mean(dim=0, keepdim=True)
        class_text_embedding /= class_text_embedding.norm(dim=1, keepdim=True)
        avg_race_4_embeddings.append(class_text_embedding)

        template_texts = [template.format(dataset_classnames_aug[label]) for template in gender_templates]
        if label == 0:
            print("gender", template_texts[0])
        template_texts = clip.tokenize(template_texts).to(device)
        class_text_embedding = model.encode_text(template_texts)
        class_text_embedding = class_text_embedding.mean(dim=0, keepdim=True)
        class_text_embedding /= class_text_embedding.norm(dim=1, keepdim=True)
        avg_gender_embeddings.append(class_text_embedding)

        template_texts = [template.format(dataset_classnames_aug[label]) for template in age_templates]
        if label == 0:
            print("age", template_texts[0])
        template_texts = clip.tokenize(template_texts).to(device)
        class_text_embedding = model.encode_text(template_texts)
        class_text_embedding = class_text_embedding.mean(dim=0, keepdim=True)
        class_text_embedding /= class_text_embedding.norm(dim=1, keepdim=True)
        avg_age_embeddings.append(class_text_embedding)

    true_img_embeddings = torch.cat(true_img_embeddings)
    idenprof_profession_labels = torch.cat(idenprof_profession_labels)
    fairface_7_race_labels = torch.cat(fairface_7_race_labels)
    fairface_4_race_labels = torch.cat(fairface_4_race_labels)
    fairface_gender_labels = torch.cat(fairface_gender_labels)
    fairface_age_labels = torch.cat(fairface_age_labels)

    avg_gen_img_embeddings = torch.cat(avg_gen_img_embeddings)

    std_text_embeddings = torch.cat(std_text_embeddings)
    avg_profession_embeddings = torch.cat(avg_profession_embeddings)
    avg_race_7_embeddings = torch.cat(avg_race_7_embeddings)
    avg_race_4_embeddings = torch.cat(avg_race_4_embeddings)
    avg_gender_embeddings = torch.cat(avg_gender_embeddings)
    avg_age_embeddings = torch.cat(avg_age_embeddings)

    print("true_img_e", true_img_embeddings.shape, "prof_l", idenprof_profession_labels.shape, "race7_l", fairface_7_race_labels.shape, "race4_l", fairface_4_race_labels.shape, "gender_l", fairface_gender_labels.shape, "age_l", fairface_age_labels.shape)
    print("std_e", std_text_embeddings.shape, "prof_e", avg_profession_embeddings.shape, "race7_e", avg_race_7_embeddings.shape, "race4_e", avg_race_4_embeddings.shape, "gender_e", avg_gender_embeddings.shape, "age_e", avg_age_embeddings.shape)
    print("gen_img_e", avg_gen_img_embeddings.shape)

    return (true_img_embeddings, idenprof_profession_labels, fairface_7_race_labels, fairface_4_race_labels, fairface_gender_labels, fairface_age_labels), (std_text_embeddings, avg_profession_embeddings, avg_race_7_embeddings, avg_race_4_embeddings, avg_gender_embeddings, avg_age_embeddings), (avg_gen_img_embeddings)