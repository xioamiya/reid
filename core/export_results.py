import pandas as pd
import os
from datetime import datetime
from PyQt5.QtWidgets import QFileDialog, QMessageBox

class ResultExporter:
    def __init__(self, parent=None):
        self.parent = parent
        
    def export_to_excel(self, results, default_name=None):
        """将检索结果导出为Excel文件"""
        try:
            if not results:
                QMessageBox.warning(self.parent, '导出警告', '没有可导出的检索结果')
                return False
                
            if default_name is None:
                default_name = f'行人检索结果_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
                
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                '导出结果',
                default_name,
                'Excel Files (*.xlsx);;All Files (*)'
            )
            
            if file_path:
                # 准备数据
                data = {
                    '视频源': [r.get('video_source', '') for r in results],
                    '时间戳': [r.get('timestamp', '') for r in results],
                    '相似度': [f"{r.get('similarity', 0)*100:.2f}%" for r in results],
                    'X坐标': [r.get('position_x', 0) for r in results],
                    'Y坐标': [r.get('position_y', 0) for r in results],
                    '检索时间': [r.get('search_time', '') for r in results]
                }
                
                # 创建DataFrame并导出
                df = pd.DataFrame(data)
                df.to_excel(file_path, index=False)
                QMessageBox.information(self.parent, '导出成功', f'结果已成功导出到：\n{file_path}')
                return True
                
        except Exception as e:
            QMessageBox.critical(self.parent, '导出错误', f'导出过程中发生错误：\n{str(e)}')
            return False
            
    def export_to_csv(self, results, default_name=None):
        """将检索结果导出为CSV文件"""
        try:
            if not results:
                QMessageBox.warning(self.parent, '导出警告', '没有可导出的检索结果')
                return False
                
            if default_name is None:
                default_name = f'行人检索结果_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                
            file_path, _ = QFileDialog.getSaveFileName(
                self.parent,
                '导出结果',
                default_name,
                'CSV Files (*.csv);;All Files (*)'
            )
            
            if file_path:
                # 准备数据
                data = {
                    '视频源': [r.get('video_source', '') for r in results],
                    '时间戳': [r.get('timestamp', '') for r in results],
                    '相似度': [f"{r.get('similarity', 0)*100:.2f}%" for r in results],
                    'X坐标': [r.get('position_x', 0) for r in results],
                    'Y坐标': [r.get('position_y', 0) for r in results],
                    '检索时间': [r.get('search_time', '') for r in results]
                }
                
                # 创建DataFrame并导出
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                QMessageBox.information(self.parent, '导出成功', f'结果已成功导出到：\n{file_path}')
                return True
                
        except Exception as e:
            QMessageBox.critical(self.parent, '导出错误', f'导出过程中发生错误：\n{str(e)}')
            return False
            
    def filter_results(self, results, min_similarity=None, video_source=None, time_range=None):
        """过滤检索结果"""
        filtered = results.copy()
        
        if min_similarity is not None:
            filtered = [r for r in filtered if r.get('similarity', 0) >= min_similarity]
            
        if video_source:
            filtered = [r for r in filtered if video_source in r.get('video_source', '')]
            
        if time_range:
            start_time, end_time = time_range
            filtered = [r for r in filtered if start_time <= r.get('timestamp', '') <= end_time]
            
        return filtered
        
    def sort_results(self, results, key='similarity', reverse=True):
        """排序检索结果"""
        if key in ['similarity', 'timestamp', 'video_source']:
            return sorted(results, key=lambda x: x.get(key, ''), reverse=reverse)
        return results