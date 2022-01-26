import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator
import mayavi.mlab as mlab


def show_pts_in_box(pts_GT, pts_pred=None, pts_3=None, pts_4=None, box3d=None):
    # draw one object and its box
    # pts: n*3, box: 1, 8, 3

    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(pts_GT, fig=fig, pts_color=(1, 0, 0))
    if pts_pred is not None:
        draw_lidar(pts_pred, fig=fig, pts_color=(0, 1, 1))
    if pts_3 is not None:
        draw_lidar(pts_3, fig=fig, pts_color=(0, 1, 0))
    if pts_4 is not None:
        draw_lidar(pts_4, fig=fig, pts_color=(1, 1, 0))
    if box3d is not None:
        draw_gt_boxes3d(box3d, fig=fig, color=(1, 1, 1))

    mlab.show()


def show_lidar_with_boxes(pts, velo, box3d,
                          img_fov=False, img_width=None, img_height=None):
    ''' Show all rect points.
        Draw 3d box in point (in rect coord system) '''

    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(pts, fig=fig, pts_color=(0, 1, 1))
    # draw_lidar(velo, fig=fig, pts_color=(1, 0, 1))
    draw_gt_boxes3d(box3d, fig=fig)
    mlab.show()


def draw_gt_boxes3d(gt_boxes3d, fig, color=(1,1,1), line_width=1, draw_text=False, text_scale=(1,1,1), color_list=None):
    ''' Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: a list of RGB tuple, if not None, overwrite color.
    Returns:
        fig: updated fig
    '''
    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if color_list is not None:
            color = color_list[n]
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def draw_lidar_simple(pc, color=None):
    ''' Draw lidar points. simplest set up. '''
    fig = mlab.figure(figure=None, bgcolor=(0,0,0), fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,2]
    #draw points
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=None, mode='point', colormap = 'gnuplot', scale_factor=1, figure=fig)
    #draw origin
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)
    #draw axis
    axes=np.array([
        [2.,0.,0.,0.],
        [0.,2.,0.,0.],
        [0.,0.,2.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)
    mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    mlab.show()


def draw_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0), pts_scale=1, pts_mode='point', pts_color=None):
    ''' Draw lidar points
    Args:
        pc: numpy array (n,3) of XYZ
        color: numpy array (n) of intensity or whatever
        fig: mayavi figure handler, if None create new one otherwise will use it
    Returns:
        fig: created or used fig
    '''
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:, 2]
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], color, color=pts_color, mode=pts_mode, colormap='gnuplot',
                  scale_factor=pts_scale, figure=fig)

    # # draw origin
    # mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
    #
    # # draw axis
    # axes = np.array([
    #     [2., 0., 0., 0.],
    #     [0., 2., 0., 0.],
    #     [0., 0., 2., 0.],
    # ], dtype=np.float64)
    # mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    # mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    # mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
    #
    # # draw fov (todo: update to real sensor spec.)
    # fov = np.array([  # 45 degree
    #     [20., 20., 0., 0.],
    #     [20., -20., 0., 0.],
    # ], dtype=np.float64)
    #
    # mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
    #             figure=fig)
    # mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]], color=(1, 1, 1), tube_radius=None, line_width=1,
    #             figure=fig)
    #
    # # draw square region
    # # TOP_Y_MIN = -20
    # # TOP_Y_MAX = 20
    # # TOP_X_MIN = 0
    # # TOP_X_MAX = 40
    # TOP_Y_MIN = -2
    # TOP_Y_MAX = 2
    # TOP_X_MIN = -2
    # TOP_X_MAX = 2
    # TOP_Z_MIN = -2.0
    # TOP_Z_MAX = 0.4
    #
    # x1 = TOP_X_MIN
    # x2 = TOP_X_MAX
    # y1 = TOP_Y_MIN
    # y2 = TOP_Y_MAX
    # mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    # mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    # mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    # mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    #
    # # mlab.orientation_axes()
    # mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


def visual_lidar(pts, box3d, path):
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    plt.title('point cloud')
    ax.scatter(pts[:, 2], pts[:, 0], pts[:, 1], c='b', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')

    # x_major_locator = MultipleLocator(5)
    # ax.xaxis.set_major_locator(x_major_locator)
    # # plt.xlim(-0.5, 80)
    # z_major_locator = MultipleLocator(1)
    # ax.zaxis.set_major_locator(z_major_locator)
    # y_major_locator = MultipleLocator(1)
    # ax.yaxis.set_major_locator(y_major_locator)


    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.scatter(box3d[:, 2], box3d[:, 0], box3d[:, 1], c='r', marker='.', s=5, linewidth=5, alpha=1, cmap='spectral')
    plt.show()


def visual_points_box3d(pts, box3d_old, path):
    # input numpy n*3 depth width height / width height depth
    # box3d 8*3
    pts_old = pts.copy()
    pts[:, 0] = pts_old[:, 2]
    pts[:, 1] = pts_old[:, 0]
    pts[:, 2] = pts_old[:, 1]
    box3d = box3d_old.copy()
    box3d[:, 0] = box3d_old[:, 2]
    box3d[:, 1] = box3d_old[:, 0]
    box3d[:, 2] = box3d_old[:, 1]
    # print('box3d', box3d)

    width_scale = [-30., 30.]
    depth_scale = [2., 80.]
    step = 0.1
    image_scale = np.array(
        [(width_scale[-1] - width_scale[0]) / step, (depth_scale[-1] - depth_scale[0]) / step]).astype('int64')
    # print('image_scale', image_scale)

    # filter
    val_inds = (pts[:, 0] >= depth_scale[0]) & (pts[:, 0] <= depth_scale[1])
    val_inds = val_inds & (pts[:, 1] <= width_scale[1]) & (pts[:, 1] >= width_scale[0])
    pts = pts[val_inds, :]

    # val_inds = (box3d[:, 0] >= depth_scale[0]) & (box3d[:, 0] <= depth_scale[1])
    # val_inds = val_inds & (box3d[:, 1] <= width_scale[1]) & (box3d[:, 1] >= width_scale[0])
    # box3d = box3d[val_inds, :]

    # print('pts', pts.shape)

    # Normalization
    pts[:, 1] = (pts[:, 1] - width_scale[0]) / step
    pts[:, 0] = (pts[:, 0] - depth_scale[0]) / step
    pts = pts.astype('int64')

    box3d[:, 1] = (box3d[:, 1] - width_scale[0]) / step
    box3d[:, 0] = (box3d[:, 0] - depth_scale[0]) / step
    box3d = box3d.astype('int64')
    # print('box3d', box3d)

    BEV = np.zeros((image_scale[0], image_scale[1], 3), np.uint8)
    BEV[pts[:, 1], pts[:, 0]] = (255, 255, 255)
    for p in box3d:
        cv2.circle(BEV, (p[0], p[1]), color=(0, 0, 255), radius=2, thickness=-1)
    #
    # BEV[box3d[:, 1], box3d[:, 0]] = (0, 0, 255)

    cv2.imwrite(path, BEV)


def visual_points(pts, path):
    # input numpy n*3 depth width height
    pts_old = pts.copy()
    pts[:, 0] = pts_old[:, 2]
    pts[:, 1] = pts_old[:, 0]
    pts[:, 2] = pts_old[:, 1]

    width_scale = [-30., 30.]
    depth_scale = [2., 80.]
    step = 0.1
    image_scale = np.array([(width_scale[-1]-width_scale[0])/step, (depth_scale[-1]-depth_scale[0])/step]).astype('int64')
    # print('image_scale', image_scale)

    # filter
    val_inds = (pts[:, 0] >= depth_scale[0]) & (pts[:, 0] <= depth_scale[1])
    val_inds = val_inds & (pts[:, 1] <= width_scale[1]) & (pts[:, 1] >= width_scale[0])
    pts = pts[val_inds, :]
    # print('pts', pts.shape)

    # Normalization
    pts[:, 1] = (pts[:, 1] - width_scale[0]) / step
    pts[:, 0] = (pts[:, 0] - depth_scale[0]) / step
    pts = pts.astype('int64')
    # print('pts', pts.shape)

    BEV = (np.zeros((image_scale)))
    BEV[pts[:, 1], pts[:, 0]] = 255
    cv2.imwrite(path, BEV)


def visual_depth_map(depth_map, path):
    depth_map = depth_map.squeeze()
    depth_map = depth_map.cpu().numpy()
    depth_map[depth_map>40.] = 40
    depth_map = ((depth_map/40.)*255).astype(np.uint8)

    im_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)  # H W 3
    # change the invalid pixel to black color
    im_color[depth_map < 1., :] = [0, 0, 0]
    cv2.imwrite(path, im_color)


