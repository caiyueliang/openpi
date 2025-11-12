import json
import shutil
import logging
from pathlib import Path
from datasets import load_dataset

def create_libero_subset_by_tasks(dataset_dir="physical-intelligence/libero", output_dir="./libero_subset", max_tasks=2, max_episodes_per_task=3):
    """
    从LIBERO数据集中抽取指定数量的任务和每个任务的episode
    
    Args:
        output_dir: 输出目录
        max_tasks: 最大任务数量
        max_episodes_per_task: 每个任务的最大episode数量
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载数据集（不下载所有数据，只获取元数据）
        dataset = load_dataset(dataset_dir, split='train', streaming=True)
        logging.info(f"[dataset] {dataset}")

        # 由于LIBERO是streaming模式，我们需要迭代获取数据
        task_count = {}
        saved_count = 0
        
        for episode in dataset:
            if saved_count >= max_tasks * max_episodes_per_task:
                break
                
            task_name = episode.get('task_name', 'unknown_task')
            
            # 统计每个任务的数量
            if task_name not in task_count:
                task_count[task_name] = 0
            
            # 如果这个任务还没达到上限，保存这个episode
            if task_count[task_name] < max_episodes_per_task and len(task_count) <= max_tasks:
                # 保存episode数据
                episode_dir = output_path / f"task_{len(task_count)}_{task_name}" / f"episode_{task_count[task_name]}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存主要数据
                episode_data = {
                    'task_name': episode['task_name'],
                    'episode_id': episode.get('episode_id', ''),
                    'language_instruction': episode.get('language_instruction', ''),
                    'keypoints': episode.get('keypoints', []),
                    # 可以根据需要添加其他字段
                }
                
                with open(episode_dir / "episode_info.json", 'w') as f:
                    json.dump(episode_data, f, indent=2)
                
                task_count[task_name] += 1
                saved_count += 1
                
                print(f"Saved episode {task_count[task_name]} for task {task_name}")
                
        print(f"Successfully created subset with {len(task_count)} tasks and {saved_count} total episodes")
        
    except Exception as e:
        print(f"Error: {e}")

from datasets import load_dataset
import json

def create_specific_tasks_subset(dataset_dir="physical-intelligence/libero", output_dir="./libero_tasks_subset", target_tasks=None, episodes_per_task=5):
    """
    抽取特定任务的数据
    
    Args:
        target_tasks: 目标任务名称列表
        episodes_per_task: 每个任务的episode数量
    """
    if target_tasks is None:
        target_tasks = ["libero_spatial", "libero_object"]  # 示例任务
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        dataset = load_dataset(dataset_dir, split='train', streaming=True)
        
        task_counts = {task: 0 for task in target_tasks}
        saved_episodes = []
        
        for episode in dataset:
            task_name = episode.get('task_name', '')
            
            if task_name in target_tasks and task_counts[task_name] < episodes_per_task:
                # 保存简化版的episode数据
                simple_episode = {
                    'task_name': task_name,
                    'language_instruction': episode.get('language_instruction', ''),
                    'keypoints_count': len(episode.get('keypoints', [])),
                    'actions_count': len(episode.get('actions', [])),
                }
                saved_episodes.append(simple_episode)
                task_counts[task_name] += 1
                
                print(f"Saved episode for task: {task_name} ({task_counts[task_name]}/{episodes_per_task})")
            
            # 检查是否所有任务都达到数量要求
            if all(count >= episodes_per_task for count in task_counts.values()):
                break
        
        # 保存所有抽取的数据
        with open(output_path / "selected_episodes.json", 'w') as f:
            json.dump(saved_episodes, f, indent=2)
            
        print(f"Successfully saved {len(saved_episodes)} episodes from {len(target_tasks)} tasks")
        
    except Exception as e:
        print(f"Error: {e}")

# 使用示例
create_specific_tasks_subset(dataset_dir="/home/caiyueliang/dataset/libero",
                             output_dir="./libero_selected", 
                             episodes_per_task=3)


# # 使用示例
# create_libero_subset_by_tasks(dataset_dir="/home/caiyueliang/dataset/libero",
#                               output_dir="./libero_demo", 
#                               max_tasks=2, 
#                               max_episodes_per_task=3)