import sys
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_vqa import blip_vqa

class VQA:
    def __init__(self, model_path, image_size=480):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = blip_vqa(pretrained=model_path, image_size=image_size, vit='base')
        self.model.eval()
        self.model = self.model.to(self.device)

    def load_demo_image(self, image_size, img_path, device):
        raw_image = Image.open(img_path).convert('RGB')   
        w,h = raw_image.size
        transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(raw_image).unsqueeze(0).to(device)   
        return raw_image, image

    def vqa(self, img_path, question):
        raw_image, image = self.load_demo_image(image_size=480, img_path=img_path, device=self.device)        
        with torch.no_grad():
            answer = self.model(image, question, train=False, inference='generate')
            return answer[0]

if __name__=="__main__":
    if not len(sys.argv) == 3:
        print('Format: python3 vqa.py <path_to_img> <question>')
        print('Sample: python3 vqa.py sample.jpg "What is the color of the horse?"')
        
    else:
        model_path = 'checkpoints/model_base_vqa_capfilt_large.pth'
        vqa_object = VQA(model_path=model_path)
        img_path = sys.argv[1]
        question = sys.argv[2]
        answer = vqa_object.vqa(img_path, question)
        print('Question: {} | Answer: {}'.format(question, answer))