import sqlite3
import os
from datetime import datetime

class ReIDDatabase:
    def __init__(self, db_path='reid_records.db'):
        self.db_path = db_path
        self.init_database()
        self._connection = None
    
    def _get_connection(self):
        """获取数据库连接，使用连接池模式"""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
        return self._connection
    
    def _close_connection(self):
        """关闭数据库连接"""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def init_database(self):
        """初始化数据库，创建必要的表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建检索记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_source TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                position_x REAL NOT NULL,
                position_y REAL NOT NULL,
                similarity REAL NOT NULL,
                search_time TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_record(self, video_source, timestamp, frame_number, position_x, position_y, similarity):
        """添加一条检索记录"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO search_records 
                (video_source, timestamp, frame_number, position_x, position_y, similarity, search_time)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (video_source, timestamp, frame_number, position_x, position_y, 
                   similarity, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"数据库错误: {str(e)}")
        except Exception as e:
            print(f"未知错误: {str(e)}")
    
    def get_records(self, video_source=None, min_similarity=None, start_time=None, end_time=None, 
                    frame_range=None, position_range=None, sort_by='similarity', sort_order='DESC'):
        """获取检索记录，支持多种过滤条件"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            query = 'SELECT * FROM search_records WHERE 1=1'
            params = []
            
            if video_source:
                query += ' AND video_source = ?'
                params.append(video_source)
            
            if min_similarity:
                query += ' AND similarity >= ?'
                params.append(min_similarity)
            
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            
            if frame_range:
                query += ' AND frame_number BETWEEN ? AND ?'
                params.extend(frame_range)
            
            if position_range:
                (min_x, max_x), (min_y, max_y) = position_range
                query += ' AND position_x BETWEEN ? AND ? AND position_y BETWEEN ? AND ?'
                params.extend([min_x, max_x, min_y, max_y])
            
            valid_sort_fields = {'similarity', 'timestamp', 'frame_number', 'search_time'}
            sort_by = sort_by if sort_by in valid_sort_fields else 'similarity'
            sort_order = 'DESC' if sort_order.upper() not in {'ASC', 'DESC'} else sort_order.upper()
            
            query += f' ORDER BY {sort_by} {sort_order}'
            
            cursor.execute(query, params)
            records = cursor.fetchall()
            return records
        except sqlite3.Error as e:
            print(f"数据库错误: {str(e)}")
            return []
        except Exception as e:
            print(f"未知错误: {str(e)}")
            return []
    
    def export_records(self, output_path, format='csv'):
        """导出检索记录到CSV或Excel文件"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM search_records')
            records = cursor.fetchall()
            
            if format == 'csv':
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', '视频源', '时间戳', '帧号', 'X坐标', 'Y坐标', 
                                    '相似度', '检索时间'])
                    writer.writerows(records)
            elif format == 'excel':
                import pandas as pd
                df = pd.DataFrame(records, columns=['ID', '视频源', '时间戳', '帧号', 
                                                   'X坐标', 'Y坐标', '相似度', '检索时间'])
                df.to_excel(output_path, index=False)
        except sqlite3.Error as e:
            print(f"数据库错误: {str(e)}")
        except Exception as e:
            print(f"未知错误: {str(e)}")
    
    def clear_records(self):
        """清空所有检索记录"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM search_records')
            conn.commit()
        except sqlite3.Error as e:
            print(f"数据库错误: {str(e)}")
        except Exception as e:
            print(f"未知错误: {str(e)}")