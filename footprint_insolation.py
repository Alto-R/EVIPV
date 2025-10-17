


import pandas as pd
from suncalc import get_position
import numpy as np
from pybdshadow.analysis import get_timetable
from pybdshadow.utils import make_clockwise
import pyvista as pv

def generate_mesh(building,resolution = 10):
    """
    生成建筑物的3D模型
    """
    # 创建一个空的组合数据集

    combined_polydata = []
    tri_info = []
    all_points = []
    all_faces = []
    num_p = 0
    import tqdm
    for i in tqdm.tqdm(range(len(building)),desc='Generating 3D mesh'):
        row = building.iloc[i]
        polygon = row['geometry']
        height = row['height']
        building_id = row['building_id']
        roof_area = row['roof_area']
        # polygon顺时针
        polygon = make_clockwise(polygon)

        # 获取2D多边形的边界
        x, y = polygon.exterior.xy

        # 创建顶部
        base_vertices = np.column_stack([x, y, np.full(len(x), 0)])
        roof_vertices = np.column_stack([x, y, np.full(len(x), height)])

        # 创建屋顶的多边形
        face = [len(roof_vertices)] + list(range(len(roof_vertices)))
        roof = pv.PolyData(roof_vertices, faces=[face]).triangulate().subdivide_adaptive(
            max_edge_len=resolution,
            max_tri_area=(resolution**2))
        
        #combined_polydata.append(roof)
        faces_id = roof.faces.reshape(-1, 4).copy()
        faces_id[:,1:]+=num_p
        num_p+=len(roof.points)
        
        all_points.append(roof.points)
        all_faces.append(faces_id)

        tri_info+=[{
            'type': 'roof',
            'building_id': building_id,
            'entity_area': roof_area,
        }]*int(len(roof.faces)/4)

        

        for i in range(len(base_vertices)-1):
            
            # 创建墙面的顶点
            wall_vertices = np.array([
                base_vertices[i],
                base_vertices[i+1],
                roof_vertices[i+1],
                roof_vertices[i],
            ])
            wall_area = np.linalg.norm(base_vertices[i+1]-base_vertices[i])*height

            # 创建墙面的多边形
            face = [len(wall_vertices)] + list(range(len(wall_vertices)))
            wall = pv.PolyData(wall_vertices, faces=[face]).triangulate().subdivide_adaptive(
                max_edge_len=resolution,
                max_tri_area=(resolution**2))

            faces_id = wall.faces.reshape(-1, 4).copy()
            faces_id[:,1:]+=num_p
            num_p+=len(wall.points)
            all_points.append(wall.points)
            all_faces.append(faces_id)


            tri_info+=[{
                'type': 'facade',
                'facade_id':i, 
                'entity_area': wall_area,
                'building_id': building_id,
                'wall_vertices': wall_vertices,
            }]*int(len(wall.faces)/4)



            #wall.cell_data.set_array(list(range(len(tri_info),len(tri_info)+int(wall.faces.shape[0]/4))), 'cell_id')

            # 将墙面添加到组合数据集中
            combined_polydata.append(wall)

    # 现在 combined_polydata 包含了所有建筑物的3D多边形
    #combined_polydata = pv.merge(combined_polydata,merge_points=False)
    #return all_points,all_faces
    combined_polydata = pv.PolyData(np.row_stack(all_points),np.row_stack(all_faces))
    
    # 三角面信息
    tri_info = pd.DataFrame(tri_info)
    tri_info['index_tri'] = range(len(tri_info))

    coords = combined_polydata.points[combined_polydata.faces.reshape((-1, 4))[:, 1:]]

    #计算面积
    # 将coords矩阵分为三个向量，分别代表三角形的三个顶点
    x = coords[:,0]
    y = coords[:, 1]
    z = coords[:, 2]

    # 计算向量AB和AC
    vector_ab = np.column_stack((y[:, 0] - x[:, 0], y[:, 1] - x[:, 1], y[:, 2] - x[:, 2]))
    vector_ac = np.column_stack((z[:, 0] - x[:, 0], z[:, 1] - x[:, 1], z[:, 2] - x[:, 2]))

    # 计算两个向量的叉乘
    cross_product = np.cross(vector_ab, vector_ac)

    # 计算叉乘向量的模
    areas = 0.5 * np.linalg.norm(cross_product, axis=1)
    tri_info['area'] = areas
    tri_info['x'] = x.tolist()
    tri_info['y'] = y.tolist()
    tri_info['z'] = z.tolist()
    return combined_polydata, tri_info

def get_sundir(date,lon,lat):
    # obtain sun position
    sunPosition = get_position(date, lon, lat)
    azimuth = sunPosition['azimuth']
    altitude = sunPosition['altitude']

    # 将角度转换为弧度
    azimuth_rad = -azimuth+np.pi/2
    altitude_rad = altitude

    # 计算球坐标系中的向量分量
    x = np.cos(altitude_rad) * np.cos(azimuth_rad)
    y = np.cos(altitude_rad) * np.sin(azimuth_rad)
    z = np.sin(altitude_rad)

    # 太阳光的方向是从太阳指向地球，因此需要反转向量
    sun_direction = np.array([x, y, -z])

    return sun_direction

def solar_insolation(mesh,lon,lat,dates=['2022-03-01'],time_precision=3600,padding=1800,method = 1):
    '''
    计算 3D 模型的太阳辐射

    Parameters
    ----------
    mesh : pyvista.PolyData
        3D 模型
    lon : float
        经度
    lat : float
        纬度
    dates : list, optional
        日期列表, by default ['2022-03-01']
    time_precision : int, optional
        时间精度, by default 3600
    padding : int, optional
        日出和日落时间间隔, by default 1800
    method : int, optional
        计算方法, by default 1, 1: 合并所有时间计算, 2: 逐个计算

    '''
    date_times = get_timetable(lon,lat, dates=dates,precision=time_precision,padding = padding)
    date_times = list(date_times['datetime'])
    sun_direction = [-get_sundir(date,lon,lat) for date in date_times]

    # 获取mesh中每个face的center坐标
    mesh_face_centers = mesh.cell_centers().points

    if method == 1:


        # 构造面与光线追踪
        all_index_tri = []
        all_ray_origins = []
        all_ray_directions = []
        all_ray_dates = []
        for date in date_times:

            sun_direction = get_sundir(date,lon,lat)
            # 为每个三角形计算太阳照射情况，光线从远处射来
            ray_origins = mesh_face_centers-sun_direction*5e5
            all_ray_origins.append(ray_origins)
            # 光线方向
            ray_directions = np.tile(sun_direction, (len(ray_origins), 1))
            all_ray_directions.append(ray_directions)

            ray_dates = np.tile(date, (len(ray_origins), 1))
            all_ray_dates.append(ray_dates)

        all_ray_origins = np.concatenate(all_ray_origins)
        all_ray_directions = np.concatenate(all_ray_directions)
        all_ray_dates = np.concatenate(all_ray_dates)

        intersection_points, intersection_rays,intersection_cells = mesh.multi_ray_trace(all_ray_origins, all_ray_directions,first_point=True) #光线追踪

        # 整合结果
        all_ray_dates = pd.DataFrame(all_ray_dates,columns=['date'])
        all_ray_dates['intersection_rays'] = range(len(all_ray_dates))
        intersection_rays_info = pd.DataFrame({'intersection_rays':intersection_rays,'intersection_cells':intersection_cells})
        all_index_tri = pd.merge(intersection_rays_info,all_ray_dates,on='intersection_rays')[['intersection_cells','date']].drop_duplicates()
        all_index_tri.rename(columns={'intersection_cells':'index_tri'},inplace=True)
    elif method == 2:

        all_index_tri = []
        import tqdm
        for date in tqdm.tqdm(date_times):
            #print(date)

            sun_direction = get_sundir(date,lon,lat)
            # 为每个三角形计算太阳照射情况，光线从远处射来
            ray_origins = mesh_face_centers-sun_direction*5e5

            # 光线方向
            ray_directions = np.tile(sun_direction, (len(ray_origins), 1))

            intersection_points, intersection_rays,intersection_cells = mesh.multi_ray_trace(ray_origins, ray_directions,first_point=True)
            
            # 光照到的三角形的索引
            index_tri_df = pd.DataFrame({'index_tri':intersection_cells}).drop_duplicates()
            index_tri_df['date'] = date
            all_index_tri.append(index_tri_df)

        all_index_tri = pd.concat(all_index_tri)

    tri_hours = (all_index_tri.groupby('index_tri').size().rename('Hours')*time_precision/3600).reset_index()

    # 上色并保存结果
    import matplotlib.pyplot as plt
    norm = plt.Normalize(0,tri_hours['Hours'].max())
    cmap = plt.cm.plasma

    hours = tri_hours[['Hours']].drop_duplicates()
    hours['colors'] = hours['Hours'].apply(lambda r:(np.array(cmap(norm(r)))*255).astype(int))
    tri_hours = tri_hours.merge(hours,on='Hours')

    import trimesh
    faces_as_array = mesh.faces.reshape((mesh.n_faces, 4))[:, 1:]
    mesh = trimesh.Trimesh(mesh.points, faces_as_array)

    # 上色并保存结果
    mesh.unmerge_vertices()
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    mesh.visual.face_colors = (np.array(cmap(0))*255).astype(int).tolist()

    for hour in tri_hours['Hours'].drop_duplicates():
        tri_hours_subdf = tri_hours[tri_hours['Hours']==hour]
        mesh.visual.face_colors[tri_hours_subdf['index_tri'].values] = tri_hours_subdf['colors'].iloc[0]

    tri_solar = all_index_tri
    return mesh,tri_solar,tri_hours[['index_tri','Hours']]