import os
import glob
import json
import re
import random
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.stats import skew, kurtosis
from scapy.all import rdpcap
from scapy.layers.tls.all import TLS, TLSClientHello, TLS_Ext_ServerName, TLSCertificate
from scapy.layers.dns import DNS, DNSQR
from tqdm import tqdm
import hashlib
from collections import Counter
import torch
from torch_geometric.data import HeteroData
from math import log2


def safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return float(a) / float(b + eps)


def l2_norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = l2_norm(a, eps)
    b = l2_norm(b, eps)
    return float(np.dot(a, b))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (np.log(p + eps) - np.log(m + eps))).sum()
    kl_qm = (q * (np.log(q + eps) - np.log(m + eps))).sum()
    return 0.5 * (kl_pm + kl_qm)


def js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    d = js_divergence(p, q)
    return 1.0 / (1.0 + d)


def hist_feature(values: List[float], bins: int, vmin: float, vmax: float) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(bins, dtype=np.float32)
    values = np.clip(np.array(values, dtype=np.float32), vmin, vmax)
    hist, _ = np.histogram(values, bins=bins, range=(vmin, vmax))
    return hist.astype(np.float32)


def calculate_entropy(data: bytes) -> float:
    if not data:
        return 0.0

    counts = Counter(data)
    data_len = len(data)

    entropy = 0.0
    for count in counts.values():
        p = count / data_len
        entropy -= p * log2(p)

    return entropy

def approx_rtt_similarity(forward_ts: List[float], backward_ts: List[float], max_pairs: int = 200) -> float:
    if len(forward_ts) == 0 or len(backward_ts) == 0:
        return 0.0
    f = np.array(forward_ts[:max_pairs], dtype=np.float64)
    b = np.array(backward_ts[:max_pairs], dtype=np.float64)
    diffs = []
    for t in f:
        diffs.append(np.min(np.abs(b - t)))
    if len(diffs) == 0:
        return 0.0
    med = float(np.median(diffs))
    return float(np.exp(-med))


class AdvancedSessionDynamicGraphGenerator:
    def __init__(
            self,
            pcap_folder: str,
            time_window_size: float = 0.1,
            seq_length: int = 20,
            max_windows: int = 100,
            pktlen_hist_bins: int = 32,
            pktlen_hist_range: Tuple[float, float] = (0, 1500),
            iat_hist_bins: int = 32,
            iat_hist_range: Tuple[float, float] = (0.0, 1.0),
            use_attr_nodes: bool = True,
            attr_feature_dim: int = 16,
            max_pcap_files: int = 5000
    ):
        self.pcap_folder = pcap_folder
        self.time_window_size = time_window_size
        self.seq_length = seq_length
        self.max_windows = max_windows

        self.pktlen_hist_bins = pktlen_hist_bins
        self.pktlen_hist_range = pktlen_hist_range
        self.iat_hist_bins = iat_hist_bins
        self.iat_hist_range = iat_hist_range

        self.use_attr_nodes = use_attr_nodes
        self.attr_feature_dim = attr_feature_dim
        self.max_pcap_files = max_pcap_files
        self.known_vpn_ports = {500, 4500, 1194, 1701, 1723}

        self.pcap_files = glob.glob(os.path.join(pcap_folder, "*.pcap"))
        if len(self.pcap_files) > self.max_pcap_files:
            print(f"[INFO] 找到 {len(self.pcap_files)} 个PCAP文件，超过限制 {self.max_pcap_files}，将随机抽样")
            self.pcap_files = random.sample(self.pcap_files, self.max_pcap_files)
        print(f"[INFO] 将处理 {len(self.pcap_files)} 个PCAP文件于 {pcap_folder}")

    def _string_to_feature(self, s: str) -> torch.FloatTensor:
        h = hashlib.md5(s.encode()).hexdigest()
        full_seed = int(h, 16)
        seed = full_seed % (2 ** 64)
        generator = torch.Generator().manual_seed(seed)
        feature_vec = torch.randn(self.attr_feature_dim, generator=generator)
        feature_vec = feature_vec / torch.linalg.norm(feature_vec)
        return feature_vec

    def extract_packet_features(self, pkt) -> Optional[Dict]:
        try:
            features = {}
            if 'IP' not in pkt:
                return None
            ip = pkt['IP']
            features['src_ip'] = ip.src
            features['dst_ip'] = ip.dst
            features['protocol'] = ip.proto
            features['ttl'] = int(ip.ttl)
            features['ip_flags'] = int(ip.flags)

            if 'TCP' in pkt:
                transport_layer = pkt['TCP']
                features['src_port'] = int(transport_layer.sport)
                features['dst_port'] = int(transport_layer.dport)
                features['tcp_flags'] = int(transport_layer.flags)
                features['window_size'] = int(transport_layer.window)
                features['payload'] = bytes(transport_layer.payload)
            elif 'UDP' in pkt:
                transport_layer = pkt['UDP']
                features['src_port'] = int(transport_layer.sport)
                features['dst_port'] = int(transport_layer.dport)
                features['tcp_flags'] = 0
                features['window_size'] = 0
                features['payload'] = bytes(transport_layer.payload)
            else:
                return None

            features['pkt_len'] = int(len(pkt))
            features['timestamp'] = float(pkt.time)

            features['tls_ciphersuites'] = []
            if pkt.haslayer(TLSClientHello):
                cs_layer = pkt.getlayer(TLSClientHello)
                if hasattr(cs_layer, 'ciphersuites'):
                    features['tls_ciphersuites'] = list(cs_layer.ciphersuites)

            if pkt.haslayer(DNS) and pkt.haslayer(DNSQR):
                if pkt[DNS].qdcount > 0:
                    query = pkt[DNS].qd
                    features['llmnr_qname'] = query.qname.decode('utf-8', errors='ignore').rstrip('.')
                    features['llmnr_qtype'] = query.qtype

            return features
        except Exception:
            return None

    def extract_contextual_attributes(self, parsed_packets: List[Dict], all_pkts_scapy) -> Dict[str, List[str]]:
        tls_versions, sni_names, cert_subjects, cert_issuers = set(), set(), set(), set()
        tls_handshake_info = {'version': '', 'ciphers': [], 'extensions': [], 'elliptic_curves': [], 'ec_point_formats': []}
        client_hello_found = False
        for p in all_pkts_scapy:
            if not p.haslayer(TLS): continue
            if not client_hello_found and p.haslayer(TLSClientHello):
                client_hello = p[TLSClientHello]
                if hasattr(p[TLS], 'version'): tls_versions.add(str(p[TLS].version)); tls_handshake_info['version'] = str(p[TLS].version)
                if hasattr(client_hello, 'ciphersuites'): tls_handshake_info['ciphers'] = [str(c) for c in client_hello.ciphersuites]
                if hasattr(client_hello, 'ext') and client_hello.ext:
                    extensions = []
                    for ext in client_hello.ext:
                        if ext is None: continue
                        extensions.append(str(ext.type))
                        if ext.type == 10 and hasattr(ext, 'curves'):
                            tls_handshake_info['elliptic_curves'] = [str(c) for c in ext.curves]
                        elif ext.type == 11 and hasattr(ext, 'fmts'):
                            tls_handshake_info['ec_point_formats'] = [str(f) for f in ext.fmts]
                        elif isinstance(ext, TLS_Ext_ServerName) and hasattr(ext, 'servernames'):
                            for sn in ext.servernames:
                                if hasattr(sn, 'servername'): sni_names.add(sn.servername.decode("utf-8", errors="ignore"))
                    tls_handshake_info['extensions'] = extensions
                client_hello_found = True
            if p.haslayer(TLSCertificate):
                if hasattr(p[TLSCertificate], 'certs'):
                    for cert in p[TLSCertificate].certs:
                        if hasattr(cert, 'subject') and hasattr(cert.subject, 'common_name'): cert_subjects.add(cert.subject.common_name.decode("utf-8", errors="ignore"))
                        if hasattr(cert, 'issuer') and hasattr(cert.issuer, 'common_name'): cert_issuers.add(cert.issuer.common_name.decode("utf-8", errors="ignore"))

        ja3_string = ""
        if client_hello_found:
            ja3_parts = [tls_handshake_info['version'], "-".join(tls_handshake_info['ciphers']), "-".join(tls_handshake_info['extensions']), "-".join(tls_handshake_info['elliptic_curves']),
                         "-".join(tls_handshake_info['ec_point_formats'])]
            ja3_string = ",".join(ja3_parts)
        ja3_hash = hashlib.md5(ja3_string.encode()).hexdigest() if ja3_string else "NoJA3"

        attributes = {
            "tls_versions": list(tls_versions) if tls_versions else ["UnknownTLS"], "sni_names": list(sni_names) if sni_names else ["UnknownSNI"],
            "cert_subjects": list(cert_subjects) if cert_subjects else ["UnknownCertSubject"], "cert_issuers": list(cert_issuers) if cert_issuers else ["UnknownCertIssuer"],
            "ja3_hash": [ja3_hash]
        }

        llmnr_qnames, llmnr_qtypes = set(), set()
        for pkt_features in parsed_packets:
            if 'llmnr_qname' in pkt_features: llmnr_qnames.add(pkt_features['llmnr_qname'])
            if 'llmnr_qtype' in pkt_features: llmnr_qtypes.add(str(pkt_features['llmnr_qtype']))

        attributes["llmnr_qname"] = list(llmnr_qnames) if llmnr_qnames else ["NoQueryName"]
        attributes["llmnr_qtype"] = list(llmnr_qtypes) if llmnr_qtypes else ["NoQueryType"]

        return attributes

    def separate_bidirectional_flows(self, packets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        if not packets: return [], []
        first_packet_idx = -1
        for i, p in enumerate(packets):
            if 'src_port' in p: first_packet_idx = i; break
        if first_packet_idx == -1: return [], []
        first = packets[first_packet_idx]
        src_ip, dst_ip, src_port, dst_port = first['src_ip'], first['dst_ip'], first['src_port'], first['dst_port']
        fwd, bwd = [], []
        for p in packets:
            if 'src_port' not in p: continue
            if (p['src_ip'] == src_ip and p['src_port'] == src_port and p['dst_ip'] == dst_ip and p['dst_port'] == dst_port):
                fwd.append(p)
            elif (p['src_ip'] == dst_ip and p['src_port'] == dst_port and p['dst_ip'] == src_ip and p['dst_port'] == src_port):
                bwd.append(p)
        return fwd, bwd

    def create_time_windows(self, start_time: float, end_time: float) -> List[Tuple[float, float]]:
        windows, cur, cnt = [], start_time, 0
        while cur < end_time and cnt < self.max_windows:
            windows.append((cur, cur + self.time_window_size))
            cur += self.time_window_size
            cnt += 1
        return windows

    def sample_seq(self, seq: List[float], length: int) -> List[float]:
        if len(seq) <= length: return seq + [0.0] * (length - len(seq))
        n_start, n_end = int(length * 0.35), int(length * 0.35)
        n_mid = length - n_start - n_end
        if n_mid <= 0: n_start, n_end, n_mid = length // 2, length - (length // 2), 0
        start_part, end_part = seq[:n_start], seq[-n_end:]
        if n_mid > 0:
            mid_start_index = (len(seq) - n_mid) // 2
            mid_part = seq[mid_start_index: mid_start_index + n_mid]
            return start_part + mid_part + end_part
        return start_part + end_part

    def calculate_burst_features(self, packets: List[Dict], burst_threshold: float = 1.0) -> Dict:
        default_burst_features = {'burst_count': 0.0, 'avg_packets_per_burst': 0.0, 'avg_bytes_per_burst': 0.0, 'avg_burst_duration': 0.0, 'avg_silence_duration': 0.0, 'std_packets_per_burst': 0.0,
                                  'std_burst_duration': 0.0, 'std_silence_duration': 0.0}
        if len(packets) < 2: return default_burst_features
        timestamps = [p['timestamp'] for p in packets]
        pkt_lens = [p['pkt_len'] for p in packets]
        iats = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        bursts, silences, current_burst_packets = [], [], []
        if not iats: return default_burst_features
        current_burst_packets.append({'timestamp': timestamps[0], 'pkt_len': pkt_lens[0]})
        for i, iat in enumerate(iats):
            packet_index = i + 1
            current_packet = {'timestamp': timestamps[packet_index], 'pkt_len': pkt_lens[packet_index]}
            if iat <= burst_threshold:
                current_burst_packets.append(current_packet)
            else:
                if len(current_burst_packets) > 0: bursts.append(current_burst_packets)
                silences.append(iat)
                current_burst_packets = [current_packet]
        if len(current_burst_packets) > 0: bursts.append(current_burst_packets)
        if not bursts: return default_burst_features
        packets_per_burst = [len(b) for b in bursts]
        bytes_per_burst = [sum(p['pkt_len'] for p in b) for b in bursts]
        burst_durations = [b[-1]['timestamp'] - b[0]['timestamp'] for b in bursts if len(b) > 0]
        burst_durations = [d if d > 0 else 1e-9 for d in burst_durations]
        return {'burst_count': len(bursts), 'avg_packets_per_burst': np.mean(packets_per_burst) if packets_per_burst else 0.0,
                'avg_bytes_per_burst': np.mean(bytes_per_burst) if bytes_per_burst else 0.0, 'avg_burst_duration': np.mean(burst_durations) if burst_durations else 0.0,
                'avg_silence_duration': np.mean(silences) if silences else 0.0, 'std_packets_per_burst': np.std(packets_per_burst) if len(packets_per_burst) > 1 else 0.0,
                'std_burst_duration': np.std(burst_durations) if len(burst_durations) > 1 else 0.0, 'std_silence_duration': np.std(silences) if len(silences) > 1 else 0.0}

    def flow_global_features(self, packets: List[Dict]) -> Dict:
        default_features = {
            'pkt_len_seq': [0.0] * self.seq_length, 'iat_seq': [0.0] * self.seq_length, 'total_packets': 0, 'total_bytes': 0, 'mean_pkt_len': 0.0, 'std_pkt_len': 0.0, 'mean_iat': 0.0, 'std_iat': 0.0,
            'syn_count': 0, 'ack_count': 0, 'fin_count': 0, 'rst_count': 0, 'port_type': 2, 'timestamps': [], 'min_pkt_len': 0.0, 'max_pkt_len': 0.0, 'median_pkt_len': 0.0, 'min_iat': 0.0,
            'max_iat': 0.0, 'median_iat': 0.0, 'mean_window_size': 0.0, 'ciphersuite_entropy': 0.0, 'duration': 0.0, 'pkt_per_sec': 0.0, 'byte_per_sec': 0.0, 'psh_count': 0, 'pkt_len_cv': 0.0,
            'is_tcp': 0.0, 'is_udp': 0.0, 'burst_count': 0.0, 'avg_packets_per_burst': 0.0, 'avg_bytes_per_burst': 0.0, 'avg_burst_duration': 0.0, 'avg_silence_duration': 0.0,
            'std_packets_per_burst': 0.0, 'std_burst_duration': 0.0, 'std_silence_duration': 0.0, 'mean_ttl': 0.0, 'std_ttl': 0.0, 'mean_ip_flags': 0.0,
            'mean_payload_entropy': 0.0, 'std_payload_entropy': 0.0, 'max_payload_entropy': 0.0,
            'pkt_len_skew': 0.0, 'pkt_len_kurtosis': 0.0, 'small_pkt_ratio': 0.0, 'is_known_vpn_port': 0.0
        }
        if len(packets) < 1: return default_features

        packets = sorted(packets, key=lambda x: x['timestamp'])
        timestamps = [p['timestamp'] for p in packets]
        pkt_lens = [p['pkt_len'] for p in packets]
        iats = [0.0] + [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        window_sizes = [p['window_size'] for p in packets]
        ttls = [p.get('ttl', 0) for p in packets]
        ip_flags_list = [p.get('ip_flags', 0) for p in packets]

        payload_entropies = [calculate_entropy(p.get('payload', b'')) for p in packets]
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        total_packets, total_bytes = len(packets), int(np.sum(pkt_lens))
        pkt_per_sec, byte_per_sec = safe_div(total_packets, duration), safe_div(total_bytes, duration)

        all_ciphers = [cs for p in packets for cs in p['tls_ciphersuites']]
        ciphersuite_entropy = 0.0
        if all_ciphers:
            counts = Counter(all_ciphers)
            probs = [c / len(all_ciphers) for c in counts.values()]
            ciphersuite_entropy = -sum(p * np.log2(p) for p in probs)

        syn = sum(1 for p in packets if (p.get('tcp_flags', 0) & 0x02) != 0)
        ack = sum(1 for p in packets if (p.get('tcp_flags', 0) & 0x10) != 0)
        fin = sum(1 for p in packets if (p.get('tcp_flags', 0) & 0x01) != 0)
        rst = sum(1 for p in packets if (p.get('tcp_flags', 0) & 0x04) != 0)
        psh = sum(1 for p in packets if (p.get('tcp_flags', 0) & 0x08) != 0)

        src_port = packets[0]['src_port'] if packets else 0
        dst_port = packets[0]['dst_port'] if packets else 0
        port_type = 0 if src_port <= 1023 else 1 if src_port <= 49151 else 2

        is_tcp, is_udp = (1.0, 0.0) if (packets[0]['protocol'] if packets else 0) == 6 else (0.0, 1.0) if (packets[0]['protocol'] if packets else 0) == 17 else (0.0, 0.0)
        mean_pkt_len = float(np.mean(pkt_lens)) if len(pkt_lens) > 0 else 0.0
        std_pkt_len = float(np.std(pkt_lens)) if len(pkt_lens) > 1 else 0.0

        features = {
            'pkt_len_seq': self.sample_seq(pkt_lens, self.seq_length), 'iat_seq': self.sample_seq(iats, self.seq_length), 'total_packets': total_packets, 'total_bytes': total_bytes,
            'mean_pkt_len': mean_pkt_len, 'std_pkt_len': std_pkt_len, 'mean_iat': float(np.mean(iats)) if len(iats) > 0 else 0.0, 'std_iat': float(np.std(iats)) if len(iats) > 1 else 0.0,
            'syn_count': syn, 'ack_count': ack, 'fin_count': fin, 'rst_count': rst, 'port_type': port_type, 'timestamps': timestamps,
            'min_pkt_len': float(np.min(pkt_lens)) if len(pkt_lens) > 0 else 0.0, 'max_pkt_len': float(np.max(pkt_lens)) if len(pkt_lens) > 0 else 0.0,
            'median_pkt_len': float(np.median(pkt_lens)) if len(pkt_lens) > 0 else 0.0, 'min_iat': float(np.min(iats)) if len(iats) > 0 else 0.0,
            'max_iat': float(np.max(iats)) if len(iats) > 0 else 0.0, 'median_iat': float(np.median(iats)) if len(iats) > 0 else 0.0,
            'mean_window_size': float(np.mean(window_sizes)) if len(window_sizes) > 0 else 0.0, 'ciphersuite_entropy': ciphersuite_entropy, 'duration': duration, 'pkt_per_sec': pkt_per_sec,
            'byte_per_sec': byte_per_sec, 'psh_count': psh, 'pkt_len_cv': safe_div(std_pkt_len, mean_pkt_len), 'is_tcp': is_tcp, 'is_udp': is_udp,
            'mean_ttl': float(np.mean(ttls)) if ttls else 0.0, 'std_ttl': float(np.std(ttls)) if len(ttls) > 1 else 0.0, 'mean_ip_flags': float(np.mean(ip_flags_list)) if ip_flags_list else 0.0,
            'mean_payload_entropy': float(np.mean(payload_entropies)) if payload_entropies else 0.0,
            'std_payload_entropy': float(np.std(payload_entropies)) if len(payload_entropies) > 1 else 0.0,
            'max_payload_entropy': float(np.max(payload_entropies)) if payload_entropies else 0.0
        }
        burst_features = self.calculate_burst_features(packets)
        features.update(burst_features)

        if len(pkt_lens) > 2 and std_pkt_len > 1e-9:
            features['pkt_len_skew'] = float(skew(pkt_lens))
            features['pkt_len_kurtosis'] = float(kurtosis(pkt_lens))
        else:
            features['pkt_len_skew'] = 0.0
            features['pkt_len_kurtosis'] = 0.0

        if total_packets > 0:
            small_pkts = sum(1 for p_len in pkt_lens if p_len < 150)
            features['small_pkt_ratio'] = small_pkts / total_packets
        else:
            features['small_pkt_ratio'] = 0.0

        if src_port in self.known_vpn_ports or dst_port in self.known_vpn_ports:
            features['is_known_vpn_port'] = 1.0
        else:
            features['is_known_vpn_port'] = 0.0

        return features

    def window_features(self, packets: List[Dict], w_start: float, w_end: float) -> Dict:
        win_pkts = [p for p in packets if w_start <= p['timestamp'] < w_end]
        if len(win_pkts) == 0: return {'pkt_len_seq': [0.0] * self.seq_length, 'iat_seq': [0.0] * self.seq_length, 'total_packets': 0, 'total_bytes': 0, 'mean_pkt_len': 0.0, 'std_pkt_len': 0.0,
                                       'mean_iat': 0.0, 'std_iat': 0.0, 'timestamps': [], 'pkt_lens_for_hist': []}
        win_pkts = sorted(win_pkts, key=lambda x: x['timestamp'])
        timestamps, pkt_lens = [p['timestamp'] for p in win_pkts], [p['pkt_len'] for p in win_pkts]
        iats = [0.0] + [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        return {'pkt_len_seq': self.sample_seq(pkt_lens, self.seq_length), 'iat_seq': self.sample_seq(iats, self.seq_length), 'total_packets': len(win_pkts), 'total_bytes': int(np.sum(pkt_lens)),
                'mean_pkt_len': float(np.mean(pkt_lens)) if len(pkt_lens) else 0.0, 'std_pkt_len': float(np.std(pkt_lens)) if len(pkt_lens) > 1 else 0.0,
                'mean_iat': float(np.mean(iats)) if len(iats) else 0.0, 'std_iat': float(np.std(iats)) if len(iats) > 1 else 0.0, 'timestamps': timestamps, 'pkt_lens_for_hist': pkt_lens}

    def build_session_hetero_graph(
            self, forward_pkts: List[Dict], backward_pkts: List[Dict], windows: List[Tuple[float, float]], all_pkts_scapy
    ) -> HeteroData:
        data = HeteroData()
        fwd_g = self.flow_global_features(forward_pkts)
        bwd_g = self.flow_global_features(backward_pkts)

        def assemble_flow_feature(g):
            total_packets_log = np.log1p(g['total_packets'])
            total_bytes_log = np.log1p(g['total_bytes'])
            pkt_per_sec_log = np.log1p(g['pkt_per_sec'])
            byte_per_sec_log = np.log1p(g['byte_per_sec'])

            numeric_features = [
                total_packets_log, total_bytes_log, g['mean_pkt_len'], g['std_pkt_len'], g['mean_iat'], g['std_iat'], g['syn_count'], g['ack_count'], g['fin_count'], g['rst_count'], g['port_type'],
                g['min_pkt_len'], g['max_pkt_len'], g['median_pkt_len'], g['min_iat'], g['max_iat'], g['median_iat'], g['mean_window_size'], g['ciphersuite_entropy'], g['duration'], pkt_per_sec_log,
                byte_per_sec_log, g['psh_count'], g['pkt_len_cv'], g['is_tcp'], g['is_udp'],
                g['burst_count'], g['avg_packets_per_burst'], g['avg_bytes_per_burst'], g['avg_burst_duration'], g['avg_silence_duration'], g['std_packets_per_burst'], g['std_burst_duration'],
                g['std_silence_duration'],
                g['mean_ttl'], g['std_ttl'], g['mean_ip_flags'],
                g['mean_payload_entropy'],
                g['std_payload_entropy'],
                g['max_payload_entropy'],
                g['pkt_len_skew'], g['pkt_len_kurtosis'], g['small_pkt_ratio'], g['is_known_vpn_port']
            ]

            numeric_features = [float(f) for f in numeric_features]

            return np.array(g['pkt_len_seq'] + g['iat_seq'] + numeric_features, dtype=np.float32)

        data['flow'].x = torch.from_numpy(np.stack([assemble_flow_feature(fwd_g), assemble_flow_feature(bwd_g)], axis=0))
        time_feats, time_meta = [], []
        for wi, (ws, we) in enumerate(windows):
            fwd_w, bwd_w = self.window_features(forward_pkts, ws, we), self.window_features(backward_pkts, ws, we)
            tvec = np.array([fwd_w['total_packets'] + bwd_w['total_packets'], fwd_w['total_bytes'] + bwd_w['total_bytes'], np.mean([fwd_w['mean_pkt_len'], bwd_w['mean_pkt_len']]),
                             np.mean([fwd_w['std_pkt_len'], bwd_w['std_pkt_len']]), np.mean([fwd_w['mean_iat'], bwd_w['mean_iat']]), np.mean([fwd_w['std_iat'], bwd_w['std_iat']])], dtype=np.float32)
            time_feats.append(tvec)
            time_meta.append(dict(idx=wi, start=ws, end=we, fwd=fwd_w, bwd=bwd_w))
        if not time_feats: time_feats = [np.zeros(6, dtype=np.float32)]; time_meta = [dict(idx=0, start=0.0, end=0.0, fwd=self.window_features([], 0, 0), bwd=self.window_features([], 0, 0))]
        data['time'].x = torch.from_numpy(np.stack(time_feats, axis=0))
        f2t_src, f2t_dst, f2t_attr, b2t_src, b2t_dst, b2t_attr = [], [], [], [], [], []
        for wi, meta in enumerate(time_meta):
            fwd_w, bwd_w = meta['fwd'], meta['bwd']
            tot_pkts, tot_bytes = fwd_w['total_packets'] + bwd_w['total_packets'], fwd_w['total_bytes'] + bwd_w['total_bytes']
            f2t_src.append(0)
            f2t_dst.append(wi)
            f2t_attr.append([safe_div(fwd_w['total_packets'], tot_pkts), safe_div(fwd_w['total_bytes'], tot_bytes)])
            b2t_src.append(1)
            b2t_dst.append(wi)
            b2t_attr.append([safe_div(bwd_w['total_packets'], tot_pkts), safe_div(bwd_w['total_bytes'], tot_bytes)])
        data['flow', 'acts_in', 'time'].edge_index = torch.tensor([f2t_src + b2t_src, f2t_dst + b2t_dst], dtype=torch.long)
        data['flow', 'acts_in', 'time'].edge_attr = torch.tensor(f2t_attr + b2t_attr, dtype=torch.float)
        t_src, t_dst, t_attr = [], [], []
        for wi in range(len(time_meta) - 1):
            sim = cosine_sim(data['time'].x[wi].numpy(), data['time'].x[wi + 1].numpy())
            len_js_win = js_similarity(hist_feature(time_meta[wi]['fwd']['pkt_lens_for_hist'] + time_meta[wi]['bwd']['pkt_lens_for_hist'], self.pktlen_hist_bins, *self.pktlen_hist_range),
                                       hist_feature(time_meta[wi + 1]['fwd']['pkt_lens_for_hist'] + time_meta[wi + 1]['bwd']['pkt_lens_for_hist'], self.pktlen_hist_bins, *self.pktlen_hist_range))
            edge_attributes = [sim, len_js_win]
            t_src.extend([wi, wi + 1])
            t_dst.extend([wi + 1, wi])
            t_attr.extend([edge_attributes, edge_attributes])
        if not t_src: t_src, t_dst, t_attr = [0], [0], [[1.0, 1.0]]
        data['time', 'evolves_to', 'time'].edge_index = torch.tensor([t_src, t_dst], dtype=torch.long)
        data['time', 'evolves_to', 'time'].edge_attr = torch.tensor(t_attr, dtype=torch.float)

        def iat_list(pkts: List[Dict]) -> List[float]:
            if len(pkts) < 2: return [0.0]
            ts = sorted([p['timestamp'] for p in pkts])
            return [0.0] + [ts[i] - ts[i - 1] for i in range(1, len(ts))]

        len_js = js_similarity(hist_feature([p['pkt_len'] for p in forward_pkts], self.pktlen_hist_bins, *self.pktlen_hist_range),
                               hist_feature([p['pkt_len'] for p in backward_pkts], self.pktlen_hist_bins, *self.pktlen_hist_range))
        iat_js = js_similarity(hist_feature(iat_list(forward_pkts), self.iat_hist_bins, *self.iat_hist_range), hist_feature(iat_list(backward_pkts), self.iat_hist_bins, *self.iat_hist_range))
        rtt_sim = approx_rtt_similarity([p['timestamp'] for p in forward_pkts], [p['timestamp'] for p in backward_pkts])
        pkt_ratio, byte_ratio = safe_div(fwd_g['total_packets'], bwd_g['total_packets']), safe_div(fwd_g['total_bytes'], bwd_g['total_bytes'])
        log_pkt_ratio, log_byte_ratio = np.log1p(pkt_ratio), np.log1p(byte_ratio)
        fb_attr = [len_js, iat_js, rtt_sim, log_pkt_ratio, log_byte_ratio]
        data['flow', 'interacts', 'flow'].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        data['flow', 'interacts', 'flow'].edge_attr = torch.tensor([fb_attr, fb_attr], dtype=torch.float)

        if self.use_attr_nodes:
            attrs = self.extract_contextual_attributes(forward_pkts + backward_pkts, all_pkts_scapy)
            attr_x, attr_type, src, dst, edge_attr, idx = [], [], [], [], [], 0
            for category, values in attrs.items():
                for v in values:
                    feature_vector = self._string_to_feature(v)
                    attr_x.append(feature_vector)
                    attr_type.append(f"{category[:9].upper()}:{v[:30]}")
                    src.extend([0, 1])
                    dst.extend([idx, idx])
                    edge_attr.extend([[1.0], [1.0]])
                    idx += 1
            if attr_x:
                data['attr'].x = torch.stack(attr_x, dim=0)
                data['attr'].names = attr_type
                data['flow', 'uses', 'attr'].edge_index = torch.tensor([src, dst], dtype=torch.long)
                data['flow', 'uses', 'attr'].edge_attr = torch.tensor(edge_attr, dtype=torch.float)
            else:
                data['attr'].x = torch.empty(0, self.attr_feature_dim).float()
        return data

    def process_single_session(self, pcap_file: str) -> Optional[Dict]:
        try:
            pkts = rdpcap(pcap_file)
        except Exception:
            return None
        if not pkts: return None
        parsed = [feat for p in pkts if (feat := self.extract_packet_features(p)) is not None]
        if not parsed: return None
        fwd, bwd = self.separate_bidirectional_flows(parsed)
        if not fwd and not bwd: return None
        all_ts = [x['timestamp'] for x in parsed]
        if not all_ts: return None
        start_t, end_t = min(all_ts), max(all_ts)
        windows = self.create_time_windows(start_t, end_t)
        if not windows: windows = [(start_t, end_t)]
        hetero_graph = self.build_session_hetero_graph(fwd, bwd, windows, pkts)
        try:
            hetero_graph.validate()
        except Exception as e:
            print(f"\n[!] Graph validation failed for file: {pcap_file}, Error: {e}");
            return None
        return {'session_graph': hetero_graph}

    def generate_dataset(self, output_folder: str, label_idx: int) -> Dict:
        os.makedirs(output_folder, exist_ok=True)
        folder_name = os.path.basename(self.pcap_folder.rstrip('/'))
        class_name = re.sub(r'[^a-zA-Z0-9_]', '_', folder_name)
        dataset_info = {'class_name': class_name, 'label_idx': label_idx, 'samples': []}
        all_samples = []
        for pcap in tqdm(self.pcap_files, desc=f"[CLASS {label_idx}] 处理PCAP文件", unit="file"):
            result = self.process_single_session(pcap)
            if result is None: continue
            sample = {'session_graph': result['session_graph'], 'label': label_idx}
            all_samples.append(sample)
            num_windows = result['session_graph']['time'].num_nodes if 'time' in result['session_graph'].node_types else 0
            dataset_info['samples'].append({'original_file': os.path.basename(pcap), 'num_windows': num_windows})
        dataset_path = os.path.join(output_folder, f"hetero_dataset_{label_idx:02d}.pt")
        torch.save(all_samples, dataset_path)
        print(f"[OK] 已保存 {len(all_samples)} 个样本到: {dataset_path}")
        info_path = os.path.join(output_folder, f"dataset_info_{label_idx:02d}.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        print(f"[OK] 数据集信息: {info_path}")
        return dataset_info


if __name__ == "__main__":
    DATASET_FOLDERS = []

    OUTPUT_ROOT = ""
    global_label_idx = 0
    for dataset_idx, dataset_folder in enumerate(DATASET_FOLDERS):
        print(f"\n===== 开始处理第 {dataset_idx + 1}/{len(DATASET_FOLDERS)} 个数据集: {dataset_folder} =====")
        class_folders = sorted([f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))])
        for class_folder in class_folders:
            class_path = os.path.join(dataset_folder, class_folder)
            print(f"\n[类别处理] 类别: {class_folder} (全局标签={global_label_idx})")
            gen = AdvancedSessionDynamicGraphGenerator(
                pcap_folder=class_path, time_window_size=0.1, seq_length=20, max_windows=10,
                use_attr_nodes=True, attr_feature_dim=16, max_pcap_files=5000)

            out_dir = os.path.join(OUTPUT_ROOT, f"class_{global_label_idx:02d}")
            gen.generate_dataset(out_dir, global_label_idx)
            global_label_idx += 1
    print(f"\n===== 所有数据集处理完成，共生成 {global_label_idx} 个类别 =====")