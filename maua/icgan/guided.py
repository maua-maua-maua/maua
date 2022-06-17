from icgan import *


def checkin(i, best_ind, total_losses, losses, regs, out, noise=None, emb=None, probs=None):
    global sample_num, hist
    name = None
    if save_every and i % save_every == 0:
        name = "output/frame_%05d.jpg" % sample_num
    pil_image = save(out, name)
    vals0 = [
        sample_num,
        i,
        total_losses[best_ind],
        losses[best_ind],
        regs[best_ind],
        np.mean(total_losses),
        np.mean(losses),
        np.mean(regs),
        np.std(total_losses),
        np.std(losses),
        np.std(regs),
    ]
    stats = (
        "sample=%d iter=%d best: total=%.2f cos=%.2f reg=%.3f avg: total=%.2f cos=%.2f reg=%.3f std: total=%.2f cos=%.2f reg=%.3f"
        % tuple(vals0)
    )
    vals1 = []
    if noise is not None:
        vals1 = [np.mean(noise), np.std(noise)]
        stats += " noise: avg=%.2f std=%.3f" % tuple(vals1)
    vals2 = []
    if emb is not None:
        vals2 = [emb.mean(), emb.std()]
        stats += " emb: avg=%.2f std=%.3f" % tuple(vals2)
    elif probs:
        best = probs[best_ind]
        inds = np.argsort(best)[::-1]
        probs = np.array(probs)
        vals2 = [
            ind2name[inds[0]],
            best[inds[0]],
            ind2name[inds[1]],
            best[inds[1]],
            ind2name[inds[2]],
            best[inds[2]],
            np.sum(probs >= 0.5) / pop_size,
            np.sum(probs >= 0.3) / pop_size,
            np.sum(probs >= 0.1) / pop_size,
        ]
        stats += " 1st=%s(%.2f) 2nd=%s(%.2f) 3rd=%s(%.2f) components: >=0.5:%.0f, >=0.3:%.0f, >=0.1:%.0f" % tuple(vals2)
    hist.append(vals0 + vals1 + vals2)
    print(stats)
    sample_num += 1


def icgan_clip():

    # @title Generate images with IC-GAN + CLIP!
    # @markdown 1. For **prompt** OpenAI suggest to use the template "A photo of a X." or "A photo of a X, a type of Y." [[paper]](https://cdn.openai.com/papers/Learning_Transferable_Visual_Models_From_Natural_Language_Supervision.pdf)
    # @markdown 1. Select type of IC-GAN model with **gen_model**: "icgan" is conditioned on an instance; "cc_icgan" is conditioned on both instance and a class index.
    # @markdown 1. Select which instance to condition on, following one of the following options:
    # @markdown     1. **input_image_instance** is the path to an input image, from either the mounted Google Drive or a manually uploaded image to "Files" (left part of the screen).
    # @markdown     1. **input_feature_index** write an integer from 0 to 1000. This will change the instance conditioning and therefore the style and semantics of the generated images. This will select one of the 1000 instance features pre-selected from ImageNet using k-means.
    # @markdown 1. For **class_index** (only valid for gen_model="cc_icgan") write an integer from 0 to 1000. This will change the ImageNet class to condition on. Consult [this link](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) for a correspondence between class name and indexes.
    # @markdown 1. Vary **truncation** from 0 to 1 to apply the [truncation trick](https://arxiv.org/abs/1809.11096). Truncation=1 will provide more diverse but possibly poorer quality images. Trucation values between 0.7 and 0.9 seem to empirically work well.
    # @markdown 4. **seed**=0 means no seed.
    prompt = "A solarpunk city scape filled with beautiful trees trending on ArtStation"  # @param {type:'string'}
    gen_model = "icgan"  # @param ['icgan', 'cc_icgan']
    if gen_model == "icgan":
        experiment_name = "icgan_biggan_imagenet_res256_nofeataug"
    else:
        experiment_name = "cc_icgan_biggan_imagenet_res256_nofeataug"
    # last_gen_model = experiment_name
    size = "256"
    input_image_instance = "solarpunk15.jpg"  # @param {type:"string"}

    input_feature_index = 500  # @param {type:'integer'}
    class_index = 627  # @param {type:'integer'} (only with cc_icgan)
    download_image = True  # @param {type:'boolean'}
    download_video = True  # @param {type:'boolean'}
    truncation = 0.85  # @param {type:'number'}
    stochastic_truncation = True  # @param {type:'boolean'}
    optimizer = "CMA-ES"  # @param ['SGD','Adam','CMA-ES','CMA-ES + SGD interleaved','CMA-ES + Adam interleaved','CMA-ES + terminal SGD','CMA-ES + terminal Adam']
    pop_size = 50  # @param {type:'integer'}
    clip_model = "ViT-B/32"  # @param ['ViT-B/32','RN50','RN101','RN50x4']
    augmentations = 64  # @param {type:'integer'}
    learning_rate = 0.1  # @param {type:'number'}
    noise_normality_loss = 0  # @param {type:'number'}
    minimum_entropy_loss = 0.0001  # @param {type:'number'}
    total_variation_loss = 0.1  # @param {type:'number'}
    iterations = 100  # @param {type:'integer'}
    terminal_iterations = 100  # @param {type:'integer'}
    show_every = 1  # @param {type:'integer'}
    save_every = 1  # @param {type:'integer'}
    fps = 2  # @param {type:'number'}
    freeze_secs = 0  # @param {type:'number'}
    seed = 10  # @param {type:'number'}
    if seed == 0:
        seed = None

    softmax_temp = 1
    emb_factor = 0.067  # calculated empirically
    loss_factor = 100
    sigma0 = 0.5  # http://cma.gforge.inria.fr/cmaes_sourcecode_page.html#practical
    cma_adapt = True
    cma_diag = False
    cma_active = True
    cma_elitist = False
    noise_size = 128
    class_size = 1000
    channels = 3
    if gen_model == "icgan":
        class_index = None

    import numpy as np

    state = None if not seed else np.random.RandomState(seed)
    np.random.seed(seed)
    # Load features
    if input_image_instance not in ["None", ""]:
        print("Obtaining instance features from input image!")
        input_feature_index = None
        feature_extractor, last_feature_extractor = load_feature_extractor(
            gen_model, last_feature_extractor, feature_extractor
        )
        input_image_tensor = preprocess_input_image(input_image_instance, int(size))
        input_features, _ = feature_extractor(input_image_tensor.cuda())
        input_features /= torch.linalg.norm(input_features, dim=-1, keepdims=True)
    elif input_feature_index is not None:
        print("Selecting an instance from pre-extracted vectors!")
        feature_extractor_name = "classification" if gen_model == "cc_icgan" else "selfsupervised"
        input_features = np.load(
            "stored_instances/imagenet_res"
            + str(size)
            + "_rn50_"
            + feature_extractor_name
            + "_kmeans_k1000_instance_features.npy",
            allow_pickle=True,
        ).item()["instance_features"][input_feature_index : input_feature_index + 1]
    else:
        input_features = None

    # Load generative model
    model, last_gen_model = load_generative_model(gen_model, last_gen_model, experiment_name, model)

    # Load CLIP model
    if clip_model != last_clip_model:
        perceptor, preprocess = clip.load(clip_model)
        last_clip_model = clip_model
    clip_res = perceptor.visual.input_resolution
    sideX = sideY = int(size)
    if sideX <= clip_res and sideY <= clip_res:
        augmentations = 1
    if "CMA" not in optimizer:
        pop_size = 1

    # Prepare other variables
    name_file = "%s_%s_class_index%s_instance_index%s" % (
        gen_model,
        prompt,
        str(class_index) if class_index is not None else "None",
        str(input_feature_index) if input_feature_index is not None else "None",
    )
    requires_grad = ("SGD" in optimizer or "Adam" in optimizer) and (
        "terminal" not in optimizer or terminal_iterations > 0
    )
    total_iterations = iterations + terminal_iterations * ("terminal" in optimizer)

    replace_to_inplace_relu(model)
    replace_to_inplace_relu(perceptor)
    ind2name = {index: wn.of2ss("%08dn" % offset).lemma_names()[0] for offset, index in utils.IMAGENET.items()}
    eps = 1e-8

    # Create noise and instance vector
    noise_vector = truncnorm.rvs(
        -2 * truncation, 2 * truncation, size=(pop_size, noise_size), random_state=state
    ).astype(
        np.float32
    )  # see https://github.com/tensorflow/hub/issues/214
    noise_vector = torch.tensor(noise_vector, requires_grad=requires_grad, device="cuda")
    if input_features is not None:
        instance_vector = torch.tensor(input_features, requires_grad=False, device="cuda")
    else:
        instance_vector = None
    if class_index is not None:
        print("Conditioning on class: ", ind2name[class_index])
    if input_feature_index is not None:
        print("Conditioning on instance with index: ", input_feature_index)

    # Prepare optimizer
    if requires_grad:
        params = [noise_vector]
        if "SGD" in optimizer:
            optim = torch.optim.SGD(params, lr=learning_rate, momentum=0.9)
        else:
            optim = torch.optim.Adam(params, lr=learning_rate)

    def ascend_txt(i, grad_step=False, show_save=False):
        global global_best_loss, global_best_iteration, global_best_noise_vector, global_best_class_vector
        regs = []
        losses = []
        total_losses = []
        best_loss = np.inf
        global_reg = torch.tensor(0, device="cuda", dtype=torch.float32, requires_grad=grad_step)
        if noise_normality_loss:
            global_reg = global_reg + noise_normality_loss * normality_loss(noise_vector)
        global_reg = loss_factor * global_reg
        if grad_step:
            global_reg.backward()
        for j in range(pop_size):
            p_s = []
            out = get_output(
                noise_vector[j : j + 1], [class_index] if class_index is not None else None, instance_vector
            )
            for aug in range(augmentations):
                if sideX <= clip_res and sideY <= clip_res or augmentations == 1:
                    apper = out
                else:
                    size = torch.randint(int(0.7 * sideX), int(0.98 * sideX), ())
                    offsetx = torch.randint(0, sideX - size, ())
                    offsety = torch.randint(0, sideX - size, ())
                    apper = out[:, :, offsetx : offsetx + size, offsety : offsety + size]
                apper = (apper + 1) / 2
                apper = nn.functional.interpolate(apper, clip_res, mode="bilinear")
                # apper = apper.clamp(0,1)
                p_s.append(apper)
            into = nom(torch.cat(p_s, 0))
            predict_clip = perceptor.encode_image(into)
            loss = loss_factor * (1 - torch.cosine_similarity(predict_clip, target_clip).mean())
            total_loss = loss
            regs.append(global_reg.item())

            with torch.no_grad():
                losses.append(loss.item())
                total_losses.append(total_loss.item() + global_reg.item())
            if total_losses[-1] < best_loss:
                best_loss = total_losses[-1]
                best_ind = j
                best_out = out
                if best_loss < global_best_loss:
                    global_best_loss = best_loss
                    global_best_iteration = i
                    with torch.no_grad():
                        global_best_noise_vector = noise_vector[best_ind]
            if grad_step:
                total_loss.backward()

        if grad_step:
            optim.step()
            optim.zero_grad()

        if show_save and (save_every and i % save_every == 0 or show_every and i % show_every == 0):
            noise = None
            emb = None
            with torch.no_grad():
                noise = noise_vector.cpu().numpy()
            checkin(i, best_ind, total_losses, losses, regs, best_out, noise, emb)
        return total_losses, best_ind

    # Obtain target CLIP representation
    tx = clip.tokenize(prompt)
    with torch.no_grad():
        target_clip = perceptor.encode_text(tx.cuda())

    global_best_loss = np.inf
    global_best_iteration = 0
    global_best_noise_vector = None

    nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if "CMA" in optimizer:
        initial_vector = np.zeros(noise_size)
        bounds = None
        cma_opts = {
            "popsize": pop_size,
            "seed": np.nan,
            "AdaptSigma": cma_adapt,
            "CMA_diagonal": cma_diag,
            "CMA_active": cma_active,
            "CMA_elitist": cma_elitist,
            "bounds": bounds,
        }
        cmaes = cma.CMAEvolutionStrategy(initial_vector, sigma0, inopts=cma_opts)

    sample_num = 0
    # machine = !nvidia-smi -L
    start = time()

    # Start noise vector optimization
    for i in range(total_iterations):
        if "CMA" in optimizer and i < iterations:
            with torch.no_grad():
                cma_results = torch.tensor(cmaes.ask(), dtype=torch.float32).cuda()
                noise_vector.data = cma_results
        if requires_grad and ("terminal" not in optimizer or i >= iterations):
            losses, best_ind = ascend_txt(i, grad_step=True, show_save="CMA" not in optimizer or i >= iterations)
            assert (
                noise_vector.requires_grad
                and noise_vector.is_leaf
                and (not optimize_class or class_vector.requires_grad and class_vector.is_leaf)
            ), (noise_vector.requires_grad, noise_vector.is_leaf, class_vector.requires_grad, class_vector.is_leaf)
        if "CMA" in optimizer and i < iterations:
            with torch.no_grad():
                losses, best_ind = ascend_txt(i, show_save=True)
                if i < iterations - 1:
                    vectors = noise_vector
                    cmaes.tell(vectors.cpu().numpy(), losses)
                elif "terminal" in optimizer and terminal_iterations:
                    pop_size = 1
                    noise_vector[0] = global_best_noise_vector
        if save_every and i % save_every == 0 or show_every and i % show_every == 0:
            print(
                "took: %d secs (%.2f sec/iter) CUDA memory: %.1f GB"
                % (time() - start, (time() - start) / (i + 1), torch.cuda.max_memory_allocated() / 1024**3)
            )

    # Obtain generated image with lowest loss.
    out = get_output(
        global_best_noise_vector.unsqueeze(0), [class_index] if class_index is not None else None, instance_vector
    )
    name = "%s_best_seed%i.png" % (name_file, seed if seed is not None else -1)
    pil_image = save(out, name)
    display(pil_image)
    print("best_loss=%.2f best_iter=%d" % (global_best_loss, global_best_iteration))

    if download_image:
        from google.colab import files, output

        files.download(name)

    if download_video:
        out = '"%s_seed%i.mp4"' % (name_file, seed if seed is not None else -1)
        file_name = "%s_seed%i.mp4" % (name_file, seed if seed is not None else -1)

        with open("list.txt", "w") as f:
            for i in range(sample_num):
                f.write("file output/frame_%05d.jpg\n" % i)
            for j in range(int(freeze_secs * fps)):
                f.write("file output/frame_%05d.jpg\n" % i)
        # !ffmpeg -r $fps -f concat -safe 0 -i list.txt -c:v libx264 -pix_fmt yuv420p -profile:v baseline -movflags +faststart -r $fps $out -y
        with open(file_name, "rb") as f:
            data_url = "data:video/mp4;base64," + b64encode(f.read()).decode()
        display(
            HTML(
                """
      <video controls autoplay loop>
            <source src="%s" type="video/mp4">
      </video>"""
                % data_url
            )
        )

        from google.colab import files, output

        output.eval_js('new Audio("https://freesound.org/data/previews/80/80921_1022651-lq.ogg").play()')
        files.download(file_name)
