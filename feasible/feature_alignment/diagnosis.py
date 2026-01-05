"""
Feature Alignment Diagnostics Tool
特征空间对齐诊断工具 - 实际可用的 Python 实现
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class DiagnosticMetrics:
    """诊断指标"""
    overall_acceptance: float
    position_acceptance: List[float]
    dynamic_acceptance: Optional[float] = None
    static_acceptance: Optional[float] = None
    kl_divergence: Optional[float] = None
    cosine_similarity: Optional[float] = None
    top_k_overlap: Optional[float] = None


class FeatureAlignmentDiagnostics:
    """
    特征空间对齐诊断工具
    
    用法：
        diagnostics = FeatureAlignmentDiagnostics(decoder)
        results = diagnostics.diagnose(test_cases)
        diagnostics.print_report()
    """
    
    def __init__(self, decoder):
        self.decoder = decoder
        self.metrics_history = []
        self.diagnosis_result = None
    
    def collect_basic_metrics(
        self, 
        test_cases: List[Dict]
    ) -> DiagnosticMetrics:
        """
        收集基础诊断指标
        
        Args:
            test_cases: 测试用例列表，每个包含 input_ids, images 等
            
        Returns:
            DiagnosticMetrics 对象
        """
        print("收集基础指标...")
        
        all_acceptance_rates = []
        position_acceptances = [[] for _ in range(self.decoder.config.gamma)]
        
        for i, test_case in enumerate(test_cases):
            print(f"  处理测试用例 {i+1}/{len(test_cases)}", end='\r')
            
            # 重置统计
            self.decoder.reset_stats()
            
            # 生成
            _ = self.decoder.generate(
                input_ids=test_case['input_ids'],
                max_new_tokens=test_case.get('max_new_tokens', 50),
                **{k: v for k, v in test_case.items() 
                   if k not in ['input_ids', 'max_new_tokens']}
            )
            
            # 获取统计
            stats = self.decoder.get_stats()
            all_acceptance_rates.append(stats['acceptance_rate'])
            
            # 如果有位置级别的统计（需要修改 decoder 支持）
            if hasattr(stats, 'position_acceptance'):
                for pos, rate in enumerate(stats['position_acceptance']):
                    if pos < len(position_acceptances):
                        position_acceptances[pos].append(rate)
        
        print()  # 新行
        
        # 计算平均值
        overall_acceptance = np.mean(all_acceptance_rates)
        position_avg = [
            np.mean(pos_rates) if pos_rates else 0.0 
            for pos_rates in position_acceptances
        ]
        
        return DiagnosticMetrics(
            overall_acceptance=overall_acceptance,
            position_acceptance=position_avg
        )
    
    def analyze_position_pattern(
        self, 
        position_acceptance: List[float]
    ) -> Dict:
        """
        分析位置接受率模式
        
        Returns:
            包含模式分析结果的字典
        """
        if not position_acceptance or len(position_acceptance) < 2:
            return {'pattern': 'insufficient_data'}
        
        # 计算下降率
        decline_rates = []
        for i in range(len(position_acceptance) - 1):
            if position_acceptance[i] > 0:
                decline = (position_acceptance[i] - position_acceptance[i+1]) / position_acceptance[i]
                decline_rates.append(decline)
        
        avg_decline = np.mean(decline_rates) if decline_rates else 0.0
        
        # 判断模式
        first_token_rate = position_acceptance[0]
        last_token_rate = position_acceptance[-1]
        total_drop = first_token_rate - last_token_rate
        
        pattern = {
            'first_token_rate': first_token_rate,
            'last_token_rate': last_token_rate,
            'total_drop': total_drop,
            'avg_decline_per_step': avg_decline,
            'rapid_decline': False,
            'analysis': ''
        }
        
        if total_drop > 0.40 and avg_decline > 0.15:
            pattern['rapid_decline'] = True
            pattern['analysis'] = (
                "⚠️  快速下降模式：首个 token 接受率高但快速恶化\n"
                "   → 可能的特征对齐问题（首token基于输入还可以，后续累积误差）"
            )
        elif first_token_rate < 0.40:
            pattern['analysis'] = (
                "❌ 所有位置都很差\n"
                "   → 严重的特征对齐问题（即使首个token也不行）"
            )
        else:
            pattern['analysis'] = (
                "✓ 接受率下降平缓\n"
                "   → 特征对齐应该不是主要问题"
            )
        
        return pattern
    
    def analyze_by_scene_dynamics(
        self,
        test_cases: List[Dict],
        scene_labels: List[str]  # 'dynamic' or 'static'
    ) -> Dict:
        """
        按场景动态性分析接受率差异
        
        Args:
            test_cases: 测试用例
            scene_labels: 每个测试用例的场景标签
            
        Returns:
            包含动态性分析的字典
        """
        print("\n分析场景动态性影响...")
        
        dynamic_rates = []
        static_rates = []
        
        for i, (test_case, label) in enumerate(zip(test_cases, scene_labels)):
            print(f"  处理 {i+1}/{len(test_cases)}", end='\r')
            
            self.decoder.reset_stats()
            _ = self.decoder.generate(
                input_ids=test_case['input_ids'],
                max_new_tokens=test_case.get('max_new_tokens', 50),
                **{k: v for k, v in test_case.items() 
                   if k not in ['input_ids', 'max_new_tokens']}
            )
            
            stats = self.decoder.get_stats()
            
            if label.lower() == 'dynamic':
                dynamic_rates.append(stats['acceptance_rate'])
            elif label.lower() == 'static':
                static_rates.append(stats['acceptance_rate'])
        
        print()
        
        if not dynamic_rates or not static_rates:
            return {'error': 'insufficient_data_for_comparison'}
        
        dynamic_avg = np.mean(dynamic_rates)
        static_avg = np.mean(static_rates)
        gap = dynamic_avg - static_avg
        
        result = {
            'dynamic_acceptance': dynamic_avg,
            'static_acceptance': static_avg,
            'gap': gap,
            'analysis': ''
        }
        
        if gap > 0.30:
            result['analysis'] = (
                f"❌ 严重的场景相关问题（差距 {gap:.2%}）\n"
                f"   动态场景: {dynamic_avg:.2%}\n"
                f"   静态场景: {static_avg:.2%}\n"
                f"   → Event 数据在静态场景缺失，导致 draft 质量差"
            )
        elif gap > 0.15:
            result['analysis'] = (
                f"⚠️  中等场景相关问题（差距 {gap:.2%}）\n"
                f"   → Event-Video 特征对齐可能有问题"
            )
        else:
            result['analysis'] = (
                f"✓ 场景动态性影响小（差距 {gap:.2%}）\n"
                f"   → 不太像是特征对齐问题"
            )
        
        return result
    
    def compare_logits_distributions(
        self,
        draft_logits: torch.Tensor,
        target_logits: torch.Tensor,
        k: int = 10
    ) -> Dict:
        """
        比较 draft 和 target 的 logits 分布
        
        Args:
            draft_logits: Draft model 的 logits [vocab_size]
            target_logits: Target model 的 logits [vocab_size]
            k: Top-k 重叠分析的 k 值
            
        Returns:
            分布相似度指标
        """
        # 确保在 CPU 上
        if draft_logits.device.type != 'cpu':
            draft_logits = draft_logits.cpu()
        if target_logits.device.type != 'cpu':
            target_logits = target_logits.cpu()
        
        # 1. Top-k 重叠
        draft_topk = torch.topk(draft_logits, k).indices.tolist()
        target_topk = torch.topk(target_logits, k).indices.tolist()
        overlap = len(set(draft_topk) & set(target_topk))
        overlap_ratio = overlap / k
        
        # 2. KL 散度
        draft_probs = F.softmax(draft_logits, dim=-1)
        target_probs = F.softmax(target_logits, dim=-1)
        kl_div = F.kl_div(
            draft_probs.log(),
            target_probs,
            reduction='batchmean'
        ).item()
        
        # 3. Cosine 相似度
        cosine_sim = F.cosine_similarity(
            draft_logits.unsqueeze(0),
            target_logits.unsqueeze(0)
        ).item()
        
        # 分析
        if overlap_ratio > 0.7 and kl_div < 1.0 and cosine_sim > 0.8:
            analysis = "✓ Logits 分布高度相似，特征对齐良好"
        elif overlap_ratio < 0.4 or kl_div > 3.0 or cosine_sim < 0.5:
            analysis = "❌ Logits 分布差异很大，特征对齐有严重问题"
        else:
            analysis = "⚠️  Logits 分布中等相似，特征对齐有一定问题"
        
        return {
            'top_k_overlap': overlap_ratio,
            'kl_divergence': kl_div,
            'cosine_similarity': cosine_sim,
            'analysis': analysis
        }
    
    def run_comparison_experiment(
        self,
        test_case: Dict,
        baseline_draft_type: str = 'video',
        test_draft_type: str = 'event'
    ) -> Dict:
        """
        运行对比实验
        
        Args:
            test_case: 测试用例
            baseline_draft_type: 基线 draft 类型
            test_draft_type: 测试 draft 类型
            
        Returns:
            对比结果
        """
        print(f"\n运行对比实验: {baseline_draft_type} vs {test_draft_type}")
        
        # 注意：这需要你的 decoder 支持切换 draft 类型
        # 这里是示意代码
        
        results = {
            'baseline_acceptance': 0.0,
            'test_acceptance': 0.0,
            'alignment_penalty': 0.0,
            'analysis': ''
        }
        
        # TODO: 实际实现取决于你的系统架构
        # 你可能需要创建两个不同的 decoder 实例
        
        return results
    
    def diagnose(
        self,
        test_cases: List[Dict],
        scene_labels: Optional[List[str]] = None,
        run_logits_analysis: bool = True,
        run_comparison: bool = True
    ) -> Dict:
        """
        运行完整诊断流程
        
        Args:
            test_cases: 测试用例列表
            scene_labels: 场景动态性标签（可选）
            run_logits_analysis: 是否运行 logits 分析
            run_comparison: 是否运行对比实验
            
        Returns:
            完整的诊断结果
        """
        print("="*70)
        print("开始 Feature Alignment 诊断")
        print("="*70)
        
        results = {}
        
        # Step 1: 基础指标
        print("\n[1/5] 收集基础指标...")
        basic_metrics = self.collect_basic_metrics(test_cases)
        results['overall_acceptance'] = basic_metrics.overall_acceptance
        results['position_acceptance'] = basic_metrics.position_acceptance
        
        print(f"整体接受率: {basic_metrics.overall_acceptance:.2%}")
        
        # Step 2: 位置模式分析
        print("\n[2/5] 分析位置模式...")
        position_pattern = self.analyze_position_pattern(
            basic_metrics.position_acceptance
        )
        results['position_pattern'] = position_pattern
        print(position_pattern['analysis'])
        
        # Step 3: 场景动态性分析
        if scene_labels is not None:
            print("\n[3/5] 分析场景动态性...")
            dynamics_analysis = self.analyze_by_scene_dynamics(
                test_cases, scene_labels
            )
            results['dynamics_analysis'] = dynamics_analysis
            print(dynamics_analysis['analysis'])
        else:
            print("\n[3/5] 跳过场景动态性分析（未提供标签）")
            results['dynamics_analysis'] = None
        
        # Step 4: Logits 相似度（如果支持）
        if run_logits_analysis:
            print("\n[4/5] Logits 分布相似度分析...")
            print("  注意: 需要手动调用 compare_logits_distributions()")
            results['logits_analysis'] = None
        else:
            print("\n[4/5] 跳过 Logits 分析")
            results['logits_analysis'] = None
        
        # Step 5: 综合诊断
        print("\n[5/5] 综合诊断...")
        diagnosis = self.make_diagnosis(results)
        results['diagnosis'] = diagnosis
        
        self.diagnosis_result = results
        return results
    
    def make_diagnosis(self, results: Dict) -> Dict:
        """
        基于收集的指标做出综合诊断
        """
        evidence_score = 0
        max_score = 0
        evidence_details = []
        
        # 证据 1: 整体接受率
        max_score += 2
        if results['overall_acceptance'] < 0.30:
            evidence_score += 2
            evidence_details.append("整体接受率很低 (<30%)")
        elif results['overall_acceptance'] < 0.50:
            evidence_score += 1
            evidence_details.append("整体接受率偏低 (<50%)")
        
        # 证据 2: 位置模式
        max_score += 2
        if results['position_pattern'].get('rapid_decline', False):
            evidence_score += 2
            evidence_details.append("位置接受率快速下降")
        elif results['position_pattern']['first_token_rate'] < 0.40:
            evidence_score += 2
            evidence_details.append("首个 token 接受率就很低")
        
        # 证据 3: 场景动态性
        if results['dynamics_analysis'] is not None:
            max_score += 3
            gap = results['dynamics_analysis'].get('gap', 0)
            if gap > 0.30:
                evidence_score += 3
                evidence_details.append(f"动态/静态场景差距很大 ({gap:.2%})")
            elif gap > 0.15:
                evidence_score += 2
                evidence_details.append(f"动态/静态场景有明显差距 ({gap:.2%})")
        
        # 计算置信度
        confidence = evidence_score / max_score if max_score > 0 else 0.0
        
        # 做出诊断
        if confidence > 0.70:
            diagnosis_text = "特征对齐是主要问题"
            severity = "严重"
            symbol = "❌"
        elif confidence > 0.50:
            diagnosis_text = "特征对齐有显著影响"
            severity = "中等"
            symbol = "⚠️"
        else:
            diagnosis_text = "特征对齐不是主要问题"
            severity = "轻微"
            symbol = "✓"
        
        diagnosis = {
            'conclusion': diagnosis_text,
            'severity': severity,
            'confidence': confidence,
            'evidence_score': evidence_score,
            'max_score': max_score,
            'evidence_details': evidence_details
        }
        
        print(f"\n{symbol} {diagnosis_text}")
        print(f"严重程度: {severity}")
        print(f"置信度: {confidence:.2%} ({evidence_score}/{max_score})")
        if evidence_details:
            print("支持证据:")
            for detail in evidence_details:
                print(f"  - {detail}")
        
        return diagnosis
    
    def print_report(self):
        """打印诊断报告"""
        if self.diagnosis_result is None:
            print("尚未运行诊断，请先调用 diagnose()")
            return
        
        results = self.diagnosis_result
        
        print("\n" + "="*70)
        print("Feature Alignment 诊断报告")
        print("="*70)
        
        print(f"\n【整体接受率】{results['overall_acceptance']:.2%}")
        
        print(f"\n【位置接受率】")
        for i, rate in enumerate(results['position_acceptance'], 1):
            print(f"  第 {i} 个 token: {rate:.2%}")
        
        if results['dynamics_analysis'] is not None:
            dyn = results['dynamics_analysis']
            print(f"\n【场景动态性】")
            print(f"  动态场景: {dyn['dynamic_acceptance']:.2%}")
            print(f"  静态场景: {dyn['static_acceptance']:.2%}")
            print(f"  差距: {dyn['gap']:.2%}")
        
        print(f"\n【综合诊断】")
        diag = results['diagnosis']
        print(f"  结论: {diag['conclusion']}")
        print(f"  严重程度: {diag['severity']}")
        print(f"  置信度: {diag['confidence']:.2%}")
        
        print(f"\n【改进建议】")
        self.print_recommendations(diag['severity'])
        
        print("="*70)
    
    def print_recommendations(self, severity: str):
        """打印改进建议"""
        if severity == "严重":
            print("  1. 【高优先级】实施特征对齐策略")
            print("     - 对比学习（使用配对的 Event-RGB 数据）")
            print("     - 训练 Event → CLIP 特征映射层")
            print("     - 或使用 E2VID 将 Event 重建为 RGB")
            print("  2. 降低 gamma 值到 2-3")
            print("  3. 专注于动态场景测试")
            print("  4. 考虑数据增强或收集更多配对数据")
        elif severity == "中等":
            print("  1. 【中优先级】尝试轻量级对齐方法")
            print("     - 添加线性投影层")
            print("     - Fine-tune draft model 的视觉编码器")
            print("  2. 优化采样策略（降低 temperature）")
            print("  3. 调整 gamma 值实验")
            print("  4. 监控不同场景的表现")
        else:
            print("  1. 特征对齐不是主要瓶颈")
            print("  2. 建议优化其他方面：")
            print("     - Gamma 值调优")
            print("     - 采样策略优化")
            print("     - Draft model 推理速度优化")
    
    def save_report(self, filepath: str = 'alignment_diagnosis.json'):
        """保存诊断结果"""
        if self.diagnosis_result is None:
            print("尚未运行诊断")
            return
        
        # 转换为可序列化的格式
        serializable_results = {}
        for key, value in self.diagnosis_result.items():
            if isinstance(value, (list, dict, str, int, float, bool, type(None))):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"诊断结果已保存到 {filepath}")


# 使用示例
if __name__ == "__main__":
    print("Feature Alignment Diagnostics Tool")
    print("特征对齐诊断工具使用示例")
    print("="*70)
    
    print("""
使用方法：

1. 准备测试数据
   test_cases = [
       {
           'input_ids': torch.tensor([[...]]),
           'images': image_tensor,
           'max_new_tokens': 50
       },
       # 更多测试用例...
   ]

2. 可选：准备场景标签
   scene_labels = ['dynamic', 'static', 'dynamic', ...]

3. 创建诊断工具
   diagnostics = FeatureAlignmentDiagnostics(decoder)

4. 运行诊断
   results = diagnostics.diagnose(
       test_cases,
       scene_labels=scene_labels
   )

5. 查看报告
   diagnostics.print_report()
   diagnostics.save_report('diagnosis.json')

6. 进阶：Logits 分析
   # 在生成过程中获取 logits
   draft_logits = ...  # 从 draft model 获取
   target_logits = ...  # 从 target model 获取
   
   logits_metrics = diagnostics.compare_logits_distributions(
       draft_logits, target_logits, k=10
   )
   print(logits_metrics)
    """)
    
    print("\n注意事项：")
    print("1. 确保 decoder 已经正确初始化")
    print("2. 测试用例应该有代表性（覆盖不同场景）")
    print("3. 如果要分析场景动态性，需要手动标注测试用例")
    print("4. Logits 分析需要修改 decoder 以暴露中间结果")
    