
import os
import matplotlib.pyplot as plt


def get_project_path(project_name):
    """
    :param project_name: 项目名称，如pythonProject
    :return: ******/project_name
    """
    # 获取当前所在文件的路径
    cur_path = os.path.abspath(os.path.dirname(__file__))

    # 获取根目录
    return cur_path[:cur_path.find(project_name)] + project_name


def draw_img_groups(img_groups: list, imgs_every_row: int = 8, block: bool = True, show_time: int = 5):
    num_groups = len(img_groups)
    for i in range(num_groups):
        assert img_groups[i].shape[0] >= imgs_every_row
        img_groups[i] = img_groups[i].cpu().squeeze(1).detach().numpy()
    fig = plt.figure()
    gs = fig.add_gridspec(num_groups, imgs_every_row)
    for i in range(num_groups):
        for j in range(imgs_every_row):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img_groups[i][j], cmap="gray")
            ax.axis("off")
    plt.tight_layout()
    plt.show(block=block)
    if not block:
        plt.pause(show_time)
        plt.close("all")


print(get_project_path(project_name="Defense"))
