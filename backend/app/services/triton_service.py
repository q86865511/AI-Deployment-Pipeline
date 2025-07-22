import asyncio
import json
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
import os

class TritonService:
    """
    Triton推理服務器管理服務
    提供模型掛載、卸載、狀態查詢和性能統計功能
    """
    
    def __init__(self, triton_url: str = None):
        """
        初始化Triton服務
        
        Args:
            triton_url: Triton推理服務器的URL地址
        """
        if triton_url is None:
            triton_url = os.environ.get("TRITON_URL", "http://localhost:8001")
        
        # 確保URL不包含/v2後綴
        if triton_url.endswith('/v2'):
            triton_url = triton_url[:-3]
        
        self.triton_url = triton_url
        self.base_url = f"{triton_url}/v2"
        print(f"Triton服務初始化 - Base URL: {self.base_url}")
        
    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        向Triton服務器發送HTTP請求
        
        Args:
            method: HTTP方法 (GET, POST等)
            endpoint: API端點
            **kwargs: 額外的請求參數
            
        Returns:
            響應的JSON數據
        """
        url = f"{self.base_url}{endpoint}"
        print(f"Triton API請求: {method} {url}")
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 30秒超時
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, url, **kwargs) as response:
                    response_text = await response.text()
                    print(f"Triton API響應 [{response.status}]: {response_text[:200]}...")
                    
                    if response.status == 200:
                        try:
                            return await response.json() if response_text else {}
                        except json.JSONDecodeError:
                            return {"status": "success", "message": response_text}
                    elif response.status == 400:
                        # 400錯誤通常表示請求格式問題
                        raise Exception(f"請求格式錯誤 [{response.status}]: {response_text}")
                    elif response.status == 404:
                        # 404表示模型未找到
                        raise Exception(f"模型不存在 [{response.status}]: {response_text}")
                    else:
                        raise Exception(f"Triton API錯誤 [{response.status}]: {response_text}")
        except aiohttp.ClientConnectorError as e:
            error_msg = f"無法連接到Triton服務器 ({self.triton_url})，請確認服務器已啟動: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
        except asyncio.TimeoutError:
            error_msg = f"Triton API請求超時: {url}"
            print(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"請求Triton API時發生錯誤: {str(e)}"
            print(error_msg)
            raise Exception(error_msg)
    
    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """
        掛載模型到Triton服務器
        
        Args:
            model_name: 模型名稱
            
        Returns:
            掛載操作的結果
        """
        try:
            endpoint = f"/repository/models/{model_name}/load"
            result = await self._make_request("POST", endpoint)
            return {
                "success": True,
                "model_name": model_name,
                "message": "模型掛載成功",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "model_name": model_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """
        從Triton服務器卸載模型
        
        Args:
            model_name: 模型名稱
            
        Returns:
            卸載操作的結果
        """
        try:
            endpoint = f"/repository/models/{model_name}/unload"
            result = await self._make_request("POST", endpoint)
            return {
                "success": True,
                "model_name": model_name,
                "message": "模型卸載成功",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "model_name": model_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """
        獲取特定模型的狀態
        
        Args:
            model_name: 模型名稱
            
        Returns:
            模型狀態信息
        """
        try:
            endpoint = f"/models/{model_name}"
            result = await self._make_request("GET", endpoint)
            return {
                "success": True,
                "model_name": model_name,
                "status": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "model_name": model_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def check_model_ready(self, model_name: str) -> Dict[str, Any]:
        """
        檢查模型是否準備就緒
        
        Args:
            model_name: 模型名稱
            
        Returns:
            模型就緒狀態
        """
        try:
            endpoint = f"/models/{model_name}/ready"
            result = await self._make_request("GET", endpoint)
            return {
                "success": True,
                "model_name": model_name,
                "ready": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "model_name": model_name,
                "ready": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_all_models(self) -> Dict[str, Any]:
        """
        獲取所有模型的狀態
        
        Returns:
            所有模型的狀態列表
        """
        try:
            # 使用正確的Triton API端點：/v2/repository/index
            endpoint = "/repository/index"
            
            # 根據Triton文檔，repository/index需要POST請求
            result = await self._make_request("POST", endpoint, 
                                             headers={"Content-Type": "application/json"},
                                             json={})
            
            # Triton repository/index API直接返回模型列表
            models = []
            if isinstance(result, list):
                models = result
            elif isinstance(result, dict) and "models" in result:
                models = result["models"]
            
            print(f"Triton repository/index返回: {len(models)} 個模型")
            for model in models:
                print(f"  - {model.get('name', 'unknown')} (狀態: {model.get('state', 'unknown')})")
            
            return {
                "success": True,
                "models": models,
                "count": len(models),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"獲取所有模型失敗: {str(e)}")
            # 嘗試備用API端點
            try:
                print("嘗試使用備用端點 /models")
                endpoint = "/models"
                result = await self._make_request("GET", endpoint)
                
                models = []
                if isinstance(result, dict):
                    if "models" in result:
                        models = result["models"] if result["models"] else []
                    elif result:
                        models = [result]
                elif isinstance(result, list):
                    models = result
                
                return {
                    "success": True,
                    "models": models,
                    "count": len(models),
                    "timestamp": datetime.now().isoformat(),
                    "note": "使用備用API端點"
                }
            except Exception as backup_error:
                print(f"備用端點也失敗: {str(backup_error)}")
                return {
                    "success": False,
                    "models": [],
                    "count": 0,
                    "error": f"主要端點錯誤: {str(e)}, 備用端點錯誤: {str(backup_error)}",
                    "timestamp": datetime.now().isoformat()
                }
    
    async def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """
        獲取模型的性能統計信息
        
        Args:
            model_name: 模型名稱
            
        Returns:
            模型統計信息，包括推論延遲
        """
        try:
            endpoint = f"/models/{model_name}/stats"
            result = await self._make_request("GET", endpoint)
            
            # 提取推論統計信息
            stats_info = {
                "success": True,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }
            
            if "model_stats" in result and len(result["model_stats"]) > 0:
                model_stat = result["model_stats"][0]
                inference_stats = model_stat.get("inference_stats", {})
                
                # 計算平均推論時間
                success_stats = inference_stats.get("success", {})
                compute_infer_stats = inference_stats.get("compute_infer", {})
                
                # 獲取執行次數（總請求次數）
                execution_count = model_stat.get("execution_count", 0)
                
                # 總延遲時間（success統計） - 計算單張圖的平均延遲
                if success_stats.get("count", 0) > 0:
                    avg_total_time_ns = success_stats.get("ns", 0) / success_stats.get("count", 1)
                    avg_total_time_ms = avg_total_time_ns / 1_000_000  # 轉換為毫秒
                    
                    # 獲取批次大小來計算單張圖的延遲
                    batch_size = 1  # 預設批次大小
                    try:
                        # 從模型名稱解析批次大小
                        model_info = self._parse_model_name(model_name)
                        batch_size = int(model_info.get("batch_size", 1))
                    except (ValueError, TypeError):
                        batch_size = 1
                    
                    # 計算單張圖的平均延遲
                    avg_total_time_per_image_ms = avg_total_time_ms / batch_size
                else:
                    avg_total_time_ms = 0
                    avg_total_time_per_image_ms = 0
                
                # 推理延遲時間（compute_infer統計） - 也計算單張圖的延遲
                if compute_infer_stats.get("count", 0) > 0:
                    avg_infer_time_ns = compute_infer_stats.get("ns", 0) / compute_infer_stats.get("count", 1)
                    avg_infer_time_ms = avg_infer_time_ns / 1_000_000  # 轉換為毫秒
                    
                    # 同樣除以批次大小
                    try:
                        model_info = self._parse_model_name(model_name)
                        batch_size = int(model_info.get("batch_size", 1))
                    except (ValueError, TypeError):
                        batch_size = 1
                    
                    avg_infer_time_per_image_ms = avg_infer_time_ms / batch_size
                else:
                    avg_infer_time_ms = 0
                    avg_infer_time_per_image_ms = 0
                
                stats_info.update({
                    "inference_count": model_stat.get("inference_count", 0),
                    "execution_count": execution_count,  # 總請求次數
                    "last_inference": model_stat.get("last_inference", 0),  # 直接使用，已經是毫秒時間戳
                    "avg_total_inference_time_ms": round(avg_total_time_per_image_ms, 2),  # 單張圖總延遲
                    "avg_infer_time_ms": round(avg_infer_time_per_image_ms, 2),  # 單張圖純推理延遲
                    "success_count": success_stats.get("count", 0),
                    "fail_count": inference_stats.get("fail", {}).get("count", 0),
                    "raw_stats": result
                })
            else:
                stats_info.update({
                    "inference_count": 0,
                    "execution_count": 0,  # 總請求次數
                    "last_inference": 0,
                    "avg_total_inference_time_ms": 0,  # 單張圖總延遲
                    "avg_infer_time_ms": 0,  # 單張圖純推理延遲
                    "success_count": 0,
                    "fail_count": 0,
                    "raw_stats": result
                })
            
            return stats_info
            
        except Exception as e:
            return {
                "success": False,
                "model_name": model_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_loaded_models_info(self) -> List[Dict[str, Any]]:
        """
        獲取所有已掛載模型的詳細信息
        
        Returns:
            已掛載模型的詳細信息列表
        """
        try:
            # 使用repository/index獲取所有模型狀態
            repo_stats = await self.get_repository_stats()
            
            if not repo_stats["success"]:
                print(f"獲取倉庫統計失敗: {repo_stats.get('error', 'Unknown error')}")
                return []
            
            models_data = repo_stats["models"]
            if not models_data:
                print("Triton服務器中沒有模型")
                return []
            
            loaded_models = []
            
            # 遍歷所有模型
            for model in models_data:
                model_name = model.get("name", "")
                model_state = model.get("state", "")
                model_version = model.get("version", "1")
                
                if not model_name:
                    continue
                
                print(f"處理模型: {model_name} (狀態: {model_state})")
                
                # 只處理已準備就緒的模型
                if model_state == "READY":
                    # 獲取模型統計信息
                    stats = await self.get_model_stats(model_name)
                    
                    # 獲取模型配置信息
                    config = await self.get_model_config(model_name, model_version)
                    
                    # 解析模型名稱以提取信息
                    model_info = self._parse_model_name(model_name)
                    
                    # 從配置中獲取額外信息
                    platform = "unknown"
                    max_batch_size = model_info.get("batch_size", "1")  # 使用解析的批次大小作為預設
                    
                    if config["success"] and config["config"]:
                        platform = config["config"].get("platform", "unknown")
                        config_batch_size = config["config"].get("max_batch_size", None)
                        
                        # 只有當配置中的批次大小不為 None 且模型名稱包含批次信息時才使用配置中的值
                        if config_batch_size is not None:
                            # 檢查模型名稱是否包含批次信息
                            if "_batch" in model_name.lower() or any(part.startswith("batch") for part in model_name.split("_")):
                                max_batch_size = str(config_batch_size)
                            else:
                                # 對於原始模型名稱（如 "test"），保持解析得到的預設值 1
                                max_batch_size = model_info.get("batch_size", "1")
                                print(f"原始模型 {model_name} 保持批次大小為 {max_batch_size}")
                        else:
                            # 配置中沒有批次大小信息，使用解析的預設值
                            max_batch_size = model_info.get("batch_size", "1")
                    
                    # 根據平台推斷格式（如果解析時未找到）
                    if model_info.get("format") == "unknown" or model_info.get("format") == "PT":
                        # 優先從 Triton 平台信息推斷格式
                        if "tensorrt" in platform.lower() or "plan" in platform.lower():
                            model_info["format"] = "ENGINE"
                            print(f"從平台推斷格式: {model_name} -> ENGINE (平台: {platform})")
                        elif "onnx" in platform.lower():
                            model_info["format"] = "ONNX"
                            print(f"從平台推斷格式: {model_name} -> ONNX (平台: {platform})")
                        elif "pytorch" in platform.lower() or "libtorch" in platform.lower():
                            model_info["format"] = "PT"
                            print(f"從平台推斷格式: {model_name} -> PT (平台: {platform})")
                        else:
                            # 如果平台信息也無法確定，保持解析的結果
                            print(f"無法從平台推斷格式: {model_name} (平台: {platform})，保持解析結果: {model_info.get('format')}")
                    
                    print(f"最終模型信息: {model_name} -> 格式: {model_info.get('format')}, 精度: {model_info.get('precision')}, 批次: {max_batch_size}")
                    
                    loaded_model_info = {
                        "model_name": model_name,
                        "display_name": model_info.get("display_name", model_name),
                        "precision": model_info.get("precision", "FP32"),
                        "batch_size": max_batch_size,
                        "format": model_info.get("format", "unknown"),
                        "platform": platform,
                        "state": model_state,
                        "version": model_version,
                        "execution_count": 0,  # 總請求次數
                        "avg_total_time_ms": 0,
                        "avg_infer_time_ms": 0,
                        "last_inference": None
                    }
                    
                    # 添加統計信息
                    if stats["success"]:
                        loaded_model_info.update({
                            "execution_count": stats.get("execution_count", 0),  # 總請求次數
                            "avg_total_time_ms": stats.get("avg_total_inference_time_ms", 0),  # 單張圖總延遲
                            "avg_infer_time_ms": stats.get("avg_infer_time_ms", 0),  # 單張圖純推理延遲
                            "last_inference": stats.get("last_inference", None)  # 毫秒時間戳，無需轉換
                        })
                        print(f"模型 {model_name} 統計信息: 總請求次數={stats.get('execution_count', 0)}, 單張圖總延遲={stats.get('avg_total_inference_time_ms', 0)}ms, 單張圖推理延遲={stats.get('avg_infer_time_ms', 0)}ms, 最後推論={stats.get('last_inference', None)}")
                    else:
                        print(f"模型 {model_name} 統計信息獲取失敗: {stats.get('error', 'Unknown error')}")
                    
                    loaded_models.append(loaded_model_info)
                    print(f"已添加模型到列表: {model_name}")
                else:
                    print(f"模型 {model_name} 狀態為 {model_state}，跳過")
            
            print(f"找到 {len(loaded_models)} 個已掛載的模型")
            return loaded_models
            
        except Exception as e:
            print(f"獲取已掛載模型信息時發生錯誤: {str(e)}")
            return []
    
    def _parse_model_name(self, model_name: str) -> Dict[str, str]:
        """
        解析模型名稱以提取精度、批次大小等信息
        
        Args:
            model_name: 模型名稱
            
        Returns:
            解析後的模型信息
        """
        info = {
            "display_name": model_name,
            "precision": "FP32",  # 原始模型預設精度
            "batch_size": "1",    # 原始模型預設批次大小
            "format": "PT"        # 原始模型預設為 PT 格式
        }
        
        try:
            # 嘗試從模型名稱中提取信息
            # 模型名稱格式通常為: modelname_sourceid_engine_fp16_batch1_size640
            parts = model_name.split("_")
            name_lower = model_name.lower()
            
            # 檢測格式信息 - 更準確的檢測邏輯
            format_detected = False
            
            # 檢查明確的格式關鍵字
            if any(part.lower() in ["engine", "tensorrt", "trt"] for part in parts):
                info["format"] = "ENGINE"
                format_detected = True
            elif "onnx" in name_lower:
                info["format"] = "ONNX"
                format_detected = True
            elif any(part.lower() in ["pytorch", "pt", "torchscript", "libtorch"] for part in parts):
                info["format"] = "PT"
                format_detected = True
            
            # 解析精度和批次大小
            precision_found = False
            batch_found = False
            
            for i, part in enumerate(parts):
                part_lower = part.lower()
                if part_lower in ["fp16", "fp32"]:
                    info["precision"] = part.upper()
                    precision_found = True
                elif part.startswith("batch") and len(part) > 5:
                    try:
                        batch_size = int(part[5:])  # 提取batch後面的數字
                        info["batch_size"] = str(batch_size)
                        batch_found = True
                    except ValueError:
                        pass
                elif part_lower.startswith("batch") and i + 1 < len(parts):
                    # 處理 "batch" "1" 分開的情況
                    try:
                        batch_size = int(parts[i + 1])
                        info["batch_size"] = str(batch_size)
                        batch_found = True
                    except (ValueError, IndexError):
                        pass
            
            # 提取顯示名稱（去除技術後綴）
            if len(parts) > 1:
                # 清理顯示名稱，移除格式相關的後綴
                display_parts = []
                for part in parts:
                    part_lower = part.lower()
                    if (part_lower in ["engine", "onnx", "fp16", "fp32", "tensorrt", "trt", "pytorch", "pt", "torchscript"] or 
                        part.startswith("batch") or part.startswith("size") or part.isdigit()):
                        break
                    display_parts.append(part)
                
                if display_parts:
                    info["display_name"] = "_".join(display_parts)
                else:
                    # 如果沒有有效的顯示部分，使用第一部分
                    info["display_name"] = parts[0]
            
            # 對於簡單的原始模型名稱（如 "test"），強制設定預設值
            if len(parts) <= 2 and not format_detected and not precision_found and not batch_found:
                info["display_name"] = model_name
                info["format"] = "PT"
                info["precision"] = "FP32"
                info["batch_size"] = "1"  # 強制設定為 1
                print(f"檢測到原始模型: {model_name}，強制使用預設值 (PT, FP32, batch=1)")
            elif not batch_found:
                # 如果沒有找到批次大小信息，但找到了其他信息，仍然預設為 1
                info["batch_size"] = "1"
                print(f"模型 {model_name} 沒有找到批次大小信息，預設為 1")
        
        except Exception as e:
            print(f"解析模型名稱時發生錯誤: {str(e)}")
            # 發生錯誤時確保使用預設值
            info["batch_size"] = "1"
        
        return info
    
    async def health_check(self) -> Dict[str, Any]:
        """
        檢查Triton服務器健康狀態
        
        Returns:
            服務器健康狀態信息
        """
        try:
            # 使用Triton的標準健康檢查端點
            endpoint = "/health/ready"
            result = await self._make_request("GET", endpoint)
            
            return {
                "success": True,
                "healthy": True,
                "status": "ready",
                "triton_url": self.triton_url,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Triton健康檢查失敗: {str(e)}")
            return {
                "success": False,
                "healthy": False,
                "status": "error",
                "error": str(e),
                "triton_url": self.triton_url,
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_server_metadata(self) -> Dict[str, Any]:
        """
        獲取Triton服務器元數據信息
        
        Returns:
            服務器元數據
        """
        try:
            endpoint = "/metadata"
            result = await self._make_request("GET", endpoint)
            return {
                "success": True,
                "metadata": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "metadata": {},
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_repository_stats(self) -> Dict[str, Any]:
        """
        獲取模型倉庫統計信息
        
        Returns:
            倉庫統計信息
        """
        try:
            # POST請求到repository/index獲取統計信息
            endpoint = "/repository/index"
            result = await self._make_request("POST", endpoint,
                                             headers={"Content-Type": "application/json"},
                                             json={"ready": True})
            
            # 統計已準備就緒的模型
            ready_models = []
            total_models = 0
            
            if isinstance(result, list):
                total_models = len(result)
                ready_models = [model for model in result if model.get("state") == "READY"]
            
            return {
                "success": True,
                "total_models": total_models,
                "ready_models": len(ready_models),
                "models": result if isinstance(result, list) else [],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "total_models": 0,
                "ready_models": 0,
                "models": [],
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_model_config(self, model_name: str, model_version: str = "1") -> Dict[str, Any]:
        """
        獲取特定模型的配置信息
        
        Args:
            model_name: 模型名稱
            model_version: 模型版本，默認為"1"
            
        Returns:
            模型配置信息
        """
        try:
            endpoint = f"/models/{model_name}/versions/{model_version}/config"
            result = await self._make_request("GET", endpoint)
            return {
                "success": True,
                "model_name": model_name,
                "version": model_version,
                "config": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "success": False,
                "model_name": model_name,
                "version": model_version,
                "config": {},
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            } 