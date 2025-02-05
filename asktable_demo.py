from asktable import Asktable
import pandas as pd
import mysql.connector
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine
from asktable.types import (
    AnswerResponse,
    Datasource,
    DatasourceRetrieveResponse,
    Meta
)
import requests
import time

# AskTable配置
ASKTABLE_API_KEY = "ADMIN_64G9I8Q2POJDAYYTS3I7"  # 请填写您的API密钥
ASKTABLE_BASE_URL = "https://api.asktable.com/v1"  # 更新API基础URL，添加版本号

def connect_mysql(host: str = 'localhost',
                 user: str = 'root',
                 password: str = '123456',
                 database: str = 'doc_generator') -> Dict:
    """连接到MySQL数据库，返回connection和engine"""
    try:
        # 创建MySQL原生连接
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        # 创建SQLAlchemy引擎
        engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database}')
        
        return {'connection': connection, 'engine': engine}
    except Exception as e:
        raise Exception(f"数据库连接失败: {str(e)}")

def load_table_data(engine, connection: mysql.connector.MySQLConnection, 
                   table_name: str,
                   limit: Optional[int] = None) -> pd.DataFrame:
    """从MySQL表加载数据到DataFrame"""
    try:
        # 首先获取所有表名
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        cursor.close()
        
        if not tables:
            raise Exception("数据库中没有表")
        
        if table_name not in tables:
            tables_str = "\n".join(f"- {table}" for table in tables)
            raise Exception(f"表 '{table_name}' 不存在。可用的表有：\n{tables_str}")
        
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        
        # 使用SQLAlchemy引擎读取数据
        return pd.read_sql(query, engine)
    except Exception as e:
        raise Exception(f"加载表数据失败: {str(e)}")

def create_datasource(asktable_client: Asktable, table_name: str, df: pd.DataFrame) -> str:
    """创建数据源"""
    try:
        print("\n正在创建数据源...")
        
        # 首先列出现有数据源
        datasources = asktable_client.datasources.list()
        print("\n现有数据源:")
        for ds in datasources.items:
            print(f"ID: {ds.id}, Name: {ds.name}")
            # 如果已存在同名数据源，先删除
            if ds.name == f"MySQL_{table_name}":
                print(f"删除已存在的数据源: {ds.name}")
                asktable_client.datasources.delete(ds.id)
        
        # 创建新的数据源
        new_datasource = asktable_client.datasources.create(
            name=f"doc_generator",
            engine="mysql",  # 使用MySQL引擎
            access_config={
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "123456",
                "db": "doc_generator",  # 使用db而不是database
                # "db_version": "8.0"  # 添加数据库版本
            },
        )
        
        print(f"创建成功! 数据源ID: {new_datasource.id}")
        
     
        return new_datasource.id
      
    except Exception as e:
        raise Exception(f"创建数据源失败: {str(e)}")

def generate_sql(question: str, datasource_ids: List[str]) -> Dict:
    """根据自然语言问题生成SQL查询语句
    
    Args:
        question: 自然语言问题
        datasource_ids: 数据源ID列表
        
    Returns:
        Dict: 包含SQL语句和其他信息的字典
    """
    try:
        # 准备请求URL和头部
        url = f"{ASKTABLE_BASE_URL}/single-turn/q2s"
        headers = {
            "Authorization": f"Bearer {ASKTABLE_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # 准备请求体
        payload = {
            "datasource_id": datasource_ids[0],  # 使用第一个数据源ID
            "question": question,
        }
        
        # 发送POST请求
        response = requests.post(url, json=payload, headers=headers)
        
        # 如果响应状态码不是200，打印详细错误信息
        if response.status_code != 200:
            print(f"API响应: {response.text}")
            response.raise_for_status()
        
        # 解析响应
        result = response.json()
        
        # 检查响应中是否包含错误信息
        if "error" in result:
            raise Exception(f"API返回错误: {result['error']}")
        
        # 提取SQL语句和其他信息
        sql_info = {
            "sql": result.get("prepared_statement", ""),
            "parameters": result.get("parameters", {}),
            "headers": result.get("header", {})
        }
        print(sql_info)
        
        return sql_info
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"请求失败: {str(e)}")
    except Exception as e:
        raise Exception(f"生成SQL失败: {str(e)}")

def query_data(asktable_client: Asktable, datasource_id: str, question: str) -> AnswerResponse:
    """执行自然语言查询"""
    try:
        # 第一步：使用自然语言生成SQL
        sql_info = generate_sql(question, [datasource_id])
        
        if not sql_info.get('sql'):
            raise Exception("未能生成有效的SQL查询")
            
        print(f"\n您的问题: {question}")
        print(f"生成的SQL: {sql_info['sql']}")
        if sql_info['headers']:
            print("字段说明:")
            for field, desc in sql_info['headers'].items():
                print(f"- {field}: {desc}")
        
        # 第二步：执行SQL查询
        return asktable_client.answers.create(
            datasource_id=datasource_id,
            question=question,
            sql=sql_info['sql'],
            max_rows=1000
        )
    except Exception as e:
        raise Exception(f"查询失败: {str(e)}")

def format_query_result(result: AnswerResponse) -> str:
    """格式化查询结果输出"""
    try:
        if hasattr(result, 'answer'):
            return result.answer
        elif hasattr(result, 'result'):
            if isinstance(result.result, dict):
                return "\n".join(f"{k}: {v}" for k, v in result.result.items())
            elif isinstance(result.result, list):
                return "\n".join(str(item) for item in result.result)
        return str(result)
    except Exception as e:
        return f"格式化结果时出错: {str(e)}"

def main():
    try:
        # 初始化AskTable客户端
        asktable_client = Asktable(
            base_url=ASKTABLE_BASE_URL,
            api_key=ASKTABLE_API_KEY
        )
        
        # 数据库连接配置
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '123456',
            'database': 'doc_generator'
        }
        
        # 连接数据库
        print("正在连接数据库...")
        db = connect_mysql(**db_config)
        connection = db['connection']
        engine = db['engine']
        
        # 获取并显示所有表
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        cursor.close()
        
        print("\n=== 数据库中的表 ===")
        for table in tables:
            print(f"- {table}")
        
        # 获取要查询的表名
        table_name = input("\n请输入要查询的表名：")
        
        # 加载表数据
        print(f"正在加载表 {table_name} 的数据...")
        df = load_table_data(engine, connection, table_name)
        
        # 创建数据源
        datasource_id = create_datasource(asktable_client, table_name, df)
        
        # 显示表结构信息
        print("\n=== 数据表字段说明 ===")
        for column in df.columns:
            print(f"- {column}: {df[column].dtype}")
        
        while True:
            # 获取用户输入的查询
            query = input("\n请输入你的问题（输入q退出）：")
            if query.lower() == 'q':
                break
            
            try:
                # 使用AskTable的自然语言查询
                response = query_data(asktable_client, datasource_id, query)
                
                # 格式化并显示结果
                print("\n查询结果：")
                print(format_query_result(response))
                    
            except Exception as e:
                print(f"查询出错：{str(e)}")
                print("提示：请尝试用更简单或更清晰的方式描述你的问题")
        
    except Exception as e:
        print(f"程序运行出错：{str(e)}")
    finally:
        # 关闭数据库连接
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            if 'engine' in locals():
                engine.dispose()
            print("\n数据库连接已关闭")
        
        # 清理数据源
        try:
            if 'datasource_id' in locals():
                asktable_client.datasources.delete(datasource_id)
                print("数据源已清理")
        except:
            pass

if __name__ == "__main__":
    main() 