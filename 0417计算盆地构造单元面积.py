import pandas as pd
from pyproj import CRS, Transformer
from geographiclib.geodesic import Geodesic

class Beijing54AreaCalculator:
    def __init__(self):
        # 北京54坐标系参数
        self.ellipsoid = (6378245, 1/298.3)  # 克拉索夫斯基椭球
        self.false_easting = 500000          # 东偏移500km

    def parse_coord(self, x, y):
        """解析含带号的坐标"""
        x_str = f"{int(x):08d}"
        zone = int(x_str[:2])            # 前两位为3度带带号
        x_actual = int(x_str[2:]) - self.false_easting  # 去带号并减东偏移
        return zone, x_actual, y

    def transformer(self, zone):
        """创建坐标转换器"""
        central_meridian = zone * 3  # 3度带中央子午线
        return Transformer.from_crs(
            CRS(f"+proj=tmerc +lat_0=0 +lon_0={central_meridian} +x_0=0 +ellps=krass +units=m"),
            CRS("EPSG:4326")
        )

    def calculate_area(self, points):
        """计算椭球面面积"""
        geod = Geodesic(*self.ellipsoid)
        polygon = geod.Polygon()
        for lat, lon in points:
            polygon.AddPoint(lat, lon)
        return abs(polygon.Compute()[2])

def process_data(input_file, output_file):
    # 读取数据
    df = pd.read_excel(input_file)
    
    # 初始化计算器
    calculator = Beijing54AreaCalculator()
    
    results = []
    for name, group in df.groupby("name"):
        try:
            # 解析坐标
            parsed_points = [calculator.parse_coord(row.X, row.Y) for row in group.itertuples()]
            zone = parsed_points[0][0]  # 取第一个点的带号
            
            # 转换坐标
            transformer = calculator.transformer(zone)
            geo_points = [transformer.transform(x, y)[::-1] for (_,x,y) in parsed_points]
            
            # 计算面积
            area = calculator.calculate_area(geo_points)
            results.append({
                "区域名称": name,
                "面积(平方公里)": area / 1e6
            })
        except Exception as e:
            print(f"区域 {name} 计算失败: {str(e)}")
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_file, index=False)

# 执行计算
process_data("一级构造单元边界坐标.xlsx", "修正面积结果.xlsx")
