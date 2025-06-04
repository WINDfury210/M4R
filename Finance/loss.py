import re
import matplotlib.pyplot as plt

def plot_loss_from_log(input_file='loss.txt', output_img='loss_curve.png'):
    """
    从日志文件直接绘制损失曲线
    参数:
        input_file: 输入的日志文件名
        output_img: 输出的图片文件名
    """
    # 存储数据
    epochs = []
    losses = []
    
    # 读取并解析日志文件
    with open(input_file, 'r') as f:
        for line in f:
            # 匹配格式如: Epoch 2625/4000, Loss: 0.081145
            match = re.search(r'Epoch (\d+)/\d+, Loss: (\d+\.\d+)', line)
            if match:
                epoch = int(match.group(1))
                loss = float(match.group(2))
                # 更新或添加数据（自动处理重复epoch）
                if epoch in epochs:
                    index = epochs.index(epoch)
                    losses[index] = loss
                else:
                    epochs.append(epoch)
                    losses.append(loss)
    
    # 绘制曲线
    plt.figure()
    plt.plot(epochs, losses, linewidth=1, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    
    # 保存图像
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"损失曲线已保存到 {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_loss_from_log()