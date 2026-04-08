import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms

from AE_CNN import AutoencoderConv
from SOM import SelfOrganizingMap


MNIST_IMAGE_SHAPE = (1, 28, 28)
MNIST_NUM_CLASSES = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina e avalia o modelo SOM-AE no conjunto MNIST."
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Dispositivo de execucao. Em 'auto', usa CUDA se estiver disponivel.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente aleatoria para reproducibilidade.",
    )
    parser.add_argument(
        "--grid-height",
        type=int,
        default=7,
        help="Altura da grade do SOM.",
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=7,
        help="Largura da grade do SOM.",
    )
    parser.add_argument(
        "--som-images",
        type=int,
        default=100,
        help="Quantidade de imagens usadas no treinamento nao supervisionado.",
    )
    parser.add_argument(
        "--classifier-images",
        type=int,
        default=500,
        help="Quantidade de imagens usadas no treinamento supervisionado.",
    )
    parser.add_argument(
        "--selection-batch-size",
        type=int,
        default=1000,
        help="Batch size usado para selecionar as amostras balanceadas do MNIST.",
    )
    parser.add_argument(
        "--som-batch-size",
        type=int,
        default=64,
        help="Batch size do DataLoader para treinamento do SOM.",
    )
    parser.add_argument(
        "--classifier-batch-size",
        type=int,
        default=64,
        help="Batch size do DataLoader para treinamento da classificacao.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=500,
        help="Batch size do DataLoader para avaliacao no conjunto de teste.",
    )
    parser.add_argument(
        "--som-epochs",
        type=int,
        default=1000,
        help="Numero de epocas do treinamento nao supervisionado.",
    )
    parser.add_argument(
        "--classifier-epochs",
        type=int,
        default=1000,
        help="Numero de epocas do treinamento supervisionado.",
    )
    parser.add_argument(
        "--som-lr",
        type=float,
        default=0.2,
        help="Taxa de aprendizado inicial do SOM.",
    )
    parser.add_argument(
        "--som-sigma",
        type=float,
        default=2.0,
        help="Raio inicial de vizinhanca do SOM.",
    )
    parser.add_argument(
        "--classifier-lr",
        type=float,
        default=0.2,
        help="Taxa de aprendizado inicial da etapa supervisionada.",
    )
    parser.add_argument(
        "--classifier-sigma",
        type=float,
        default=2.0,
        help="Raio inicial de vizinhanca da etapa supervisionada.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint para retomar treinamento ou apenas avaliar.",
    )
    parser.add_argument(
        "--save-checkpoint",
        type=Path,
        default=Path("checkpoints/som_ae_mnist.pt"),
        help="Caminho do checkpoint salvo ao final da execucao.",
    )
    parser.add_argument(
        "--skip-som-training",
        action="store_true",
        help="Nao executa a etapa nao supervisionada.",
    )
    parser.add_argument(
        "--skip-classifier-training",
        action="store_true",
        help="Nao executa a etapa supervisionada.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Nao calcula a acuracia no conjunto de teste.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Nao exibe a grade com matplotlib ao final.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Diretorio local do conjunto MNIST.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas nao esta disponivel neste ambiente.")
    return torch.device(device_arg)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def selecionar_imagens(trainloader, shape, qtd_class, total_img_som, total_img_classif):
    if (total_img_som % qtd_class != 0) or (total_img_classif % qtd_class != 0):
        raise ValueError(
            "A quantidade de imagens para cada etapa deve ser multipla da quantidade de classes."
        )

    imagens_som = torch.empty(total_img_som, shape[0], shape[1], shape[2])
    rotulos_som = torch.empty(total_img_som, dtype=torch.long)
    imagens_classif = torch.empty(total_img_classif, shape[0], shape[1], shape[2])
    rotulos_classif = torch.empty(total_img_classif, dtype=torch.long)

    total_classe_som = total_img_som // qtd_class
    total_classe_classif = total_img_classif // qtd_class
    cont_img_som = 0
    cont_img_classif = 0
    cont_classe_som = np.zeros(qtd_class, dtype=np.int64)
    cont_classe_classif = np.zeros(qtd_class, dtype=np.int64)

    with torch.no_grad():
        for imagens, rotulos in trainloader:
            for img, rot in zip(imagens, rotulos):
                classe = rot.item()
                if cont_img_som < total_img_som:
                    if cont_classe_som[classe] == total_classe_som:
                        continue
                    imagens_som[cont_img_som] = img.clone()
                    rotulos_som[cont_img_som] = classe
                    cont_img_som += 1
                    cont_classe_som[classe] += 1
                    continue

                if cont_img_classif < total_img_classif:
                    if cont_classe_classif[classe] == total_classe_classif:
                        continue
                    imagens_classif[cont_img_classif] = img.clone()
                    rotulos_classif[cont_img_classif] = classe
                    cont_img_classif += 1
                    cont_classe_classif[classe] += 1

                if cont_img_som == total_img_som and cont_img_classif == total_img_classif:
                    return imagens_som, rotulos_som, imagens_classif, rotulos_classif

    raise RuntimeError("Nao foi possivel selecionar a quantidade balanceada de imagens solicitada.")


def build_train_dataloaders(args, device):
    transform_mnist = transforms.Compose([transforms.ToTensor()])
    trainset_mnist = datasets.MNIST(
        root=args.data_root,
        train=True,
        download=True,
        transform=transform_mnist,
    )

    source_loader = torch.utils.data.DataLoader(
        trainset_mnist,
        batch_size=args.selection_batch_size,
        shuffle=True,
    )

    imgs_treino_som, labels_treino_som, imgs_treino_classif, labels_treino_classif = selecionar_imagens(
        source_loader,
        MNIST_IMAGE_SHAPE,
        MNIST_NUM_CLASSES,
        args.som_images,
        args.classifier_images,
    )

    dataset_som = torch.utils.data.TensorDataset(
        imgs_treino_som.to(device),
        labels_treino_som.to(device),
    )
    dataset_classif = torch.utils.data.TensorDataset(
        imgs_treino_classif.to(device),
        labels_treino_classif.to(device),
    )

    trainloader_som = torch.utils.data.DataLoader(
        dataset_som,
        batch_size=args.som_batch_size,
        shuffle=True,
    )
    trainloader_classif = torch.utils.data.DataLoader(
        dataset_classif,
        batch_size=args.classifier_batch_size,
        shuffle=True,
    )
    return trainloader_som, trainloader_classif


def build_test_loader(args):
    transform_mnist = transforms.Compose([transforms.ToTensor()])
    testset_mnist = datasets.MNIST(
        root=args.data_root,
        train=False,
        download=True,
        transform=transform_mnist,
    )
    return torch.utils.data.DataLoader(
        testset_mnist,
        batch_size=args.test_batch_size,
        shuffle=False,
    )


def instantiate_model(dim_grade, device):
    return SelfOrganizingMap(
        classeAutoencoder=AutoencoderConv,
        dimGrade=dim_grade,
        dimImagem=MNIST_IMAGE_SHAPE,
        device=device,
    ).to(device)


def ensure_model_on_device(model, device):
    model = model.to(device)
    model.device = device
    if hasattr(model, "grade") and isinstance(model.grade, list):
        for row in model.grade:
            for autoencoder in row:
                autoencoder.to(device)
    return model


def load_checkpoint(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, SelfOrganizingMap):
        return ensure_model_on_device(checkpoint, device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        dim_grade = tuple(checkpoint.get("dim_grade", (7, 7)))
        dim_imagem = tuple(checkpoint.get("dim_imagem", MNIST_IMAGE_SHAPE))
        model = SelfOrganizingMap(
            classeAutoencoder=AutoencoderConv,
            dimGrade=dim_grade,
            dimImagem=dim_imagem,
            device=device,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    raise ValueError(f"Formato de checkpoint nao suportado: {path}")


def save_checkpoint(model, path, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dim_grade": tuple(model.dimGrade),
            "dim_imagem": tuple(model.dimImagem),
            "seed": args.seed,
            "som_images": args.som_images,
            "classifier_images": args.classifier_images,
            "som_epochs": args.som_epochs,
            "classifier_epochs": args.classifier_epochs,
        },
        path,
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    print(f"DEVICE: {device}")
    print(
        "CONFIG: "
        f"grade=({args.grid_height},{args.grid_width}) "
        f"som_imgs={args.som_images} "
        f"classif_imgs={args.classifier_images} "
        f"som_epocas={args.som_epochs} "
        f"classif_epocas={args.classifier_epochs}"
    )

    trainloader_som = None
    trainloader_classif = None
    if not args.skip_som_training or not args.skip_classifier_training:
        trainloader_som, trainloader_classif = build_train_dataloaders(args, device)

    if args.load_checkpoint is not None:
        print(f"Carregando checkpoint: {args.load_checkpoint}")
        som = load_checkpoint(args.load_checkpoint, device)
    else:
        som = instantiate_model((args.grid_height, args.grid_width), device)

    print(f"MODELO: grade={tuple(som.dimGrade)} dim_imagem={tuple(som.dimImagem)}")

    som.train()

    treinou_alguma_etapa = False

    if not args.skip_som_training:
        inicio = time.time()
        som.treinar(trainloader_som, args.som_epochs, args.som_lr, args.som_sigma)
        fim = time.time()
        print(f"TEMPO_SOM: {fim - inicio:.2f}s")
        treinou_alguma_etapa = True

    if not args.skip_classifier_training:
        inicio = time.time()
        som.treinarClassificacao(
            trainloader_classif,
            args.classifier_epochs,
            args.classifier_lr,
            args.classifier_sigma,
        )
        fim = time.time()
        print(f"TEMPO_CLASSIFICACAO: {fim - inicio:.2f}s")
        treinou_alguma_etapa = True

    if args.save_checkpoint is not None and treinou_alguma_etapa:
        save_checkpoint(som, args.save_checkpoint, args)
        print(f"Checkpoint salvo em: {args.save_checkpoint}")

    som.eval()

    if not args.skip_plot:
        som.exibirGrade()

    if not args.skip_eval:
        testloader_mnist = build_test_loader(args)
        acuracia = som.acuracia(testloader_mnist)
        print(f"ACURACIA: {acuracia:.4f}")


if __name__ == "__main__":
    main()
