/* ============================================================================
 * shape_based_matching 的 C# 封装 (P/Invoke 到 shape_based_matching_c.dll)
 *
 * 使用方法:
 *   1. 把 shape_based_matching_c.dll 与 opencv_world4130.dll 放进程序运行目录
 *      (两者都在 build/Release 下, 与 pybind 模块同一批产物);
 *   2. 把本文件加入工程;
 *   3. 口径与 Python 侧一致: ROI = [x, y, w, h], 图像为行优先连续内存 (灰度 1 通道 / BGR 3 通道)。
 *
 * 与 OpenCvSharp 配合: 定义编译常量 SBM_OPENCVSHARP 后会多出 FromMat() 便捷方法,
 * 否则直接用 RawImage 传指针。
 *
 * 注意: 字符串一律手工按 UTF-8 封送 —— 默认 CharSet.Ansi 会走系统代码页(GBK),
 * 中文类别名/目录会乱码。
 *
 * 自动探测训练阈值(省去逐图手调弱/强阈值):
 *     ThresholdSearchParams sp = ThresholdSearchParams.Default();
 *     sp.FeatureNum = 128;
 *     ThresholdEstimate est = matcher.EstimateThresholds(img, new[] { x, y, w, h },
 *                                                        sp, zones, null, null);
 *     if (est.Ok) {
 *         params.WeakThresh = est.WeakThresh;
 *         params.StrongThresh = est.StrongThresh;
 *     } else {
 *         // est.Note 会说明是 ROI 选得不好还是图本身没什么梯度
 *     }
 *     matcher.Train(img, new[] { x, y, w, h }, classId, params, saveDir, zones, null, null);
 * ==========================================================================*/
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace ShapeBasedMatching
{
    /// <summary>图像描述: 行优先连续内存, Channels = 1(灰度) 或 3(BGR)</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct RawImage
    {
        public IntPtr Data;
        public int Width;
        public int Height;
        public int Channels;
        public int Step;        // 每行字节数; 0 = Width*Channels
    }

    /// <summary>训练参数 (对应 sbm_train_params_t)</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct TrainParams
    {
        public int FeatureNum;
        public int PyramidLevelCount;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public int[] PyramidLevels;
        public double WeakThresh;
        public double StrongThresh;
        public double AngleStart;
        public double AngleExtent;
        public double AngleStep;
        public double ScaleStart;
        public double ScaleEnd;
        public double ScaleStep;

        /// <summary>一份与 C++ 侧一致的默认参数: feature_num=100, pyramid=[4,8], weak=30, strong=60</summary>
        public static TrainParams Default()
        {
            TrainParams p = new TrainParams();
            Native.sbm_train_params_init(ref p);
            return p;
        }
    }

    /// <summary>排除区</summary>
    public sealed class ExclusionZone
    {
        public string Type = "exclude_rect";   // "exclude_rect" / "exclude_ellipse"
        public int X, Y, W, H;
    }

    /// <summary>训练结果</summary>
    public sealed class TrainResult
    {
        public string YamlPath;
        public string InfoPath;
        public string PreviewPath;          // 预览图未写出时为 null
        public string SaveDir;
        public string BaseName;
        public int TemplateCount;
        /// <summary>特征点标注图宽 (ROI 宽)</summary>
        public int FeaturesWidth;
        /// <summary>特征点标注图高 (ROI 高)</summary>
        public int FeaturesHeight;
        /// <summary>特征点标注图通道数 (与输入一致: 1 灰度 / 3 BGR)</summary>
        public int FeaturesChannels;
        /// <summary>特征点标注图像素 (行优先, 已去掉行对齐)。
        /// 在 Train 返回前就拷进托管内存了, 句柄释放后依然可用。</summary>
        public byte[] FeaturesImage;
    }

    /// <summary>阈值自动探测参数 (对应 sbm_threshold_search_params_t)。
    /// 零值一律解释为"用默认", 拿不准就调 Default()。</summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ThresholdSearchParams
    {
        public int FeatureNum;                              // <=0 -> 100
        public int PyramidLevelCount;                      // <=0 -> [4, 8]
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
        public int[] PyramidLevels;
        public double ScaleEnd;                             // <=0 -> 1.0
        public double WeakRatio;                            // <=0 -> 0.5 (弱 = 弱比例 * 强)
        public double StrongMin;                            // <=0 -> 4.0
        public double StrongMax;                            // <=0 -> 255.0
        /// <summary>0=默认(开) / 1=强制开 / -1=强制关</summary>
        [MarshalAs(UnmanagedType.I4)]
        public int RunSelfCheck;

        /// <summary>一份与 C++ 侧一致的默认参数</summary>
        public static ThresholdSearchParams Default()
        {
            // 先把金字塔数组备好, 免得 marshal 一个 null 的 ByValArray
            ThresholdSearchParams p = new ThresholdSearchParams
            {
                PyramidLevels = new int[8],
            };
            Native.sbm_threshold_search_params_init(ref p);
            return p;
        }
    }

    /// <summary>阈值自动探测结果 (对应 sbm_threshold_estimate_t)。
    /// Ok = false 时 WeakThresh / StrongThresh 是默认值 30/60, 请直接看 Note。</summary>
    public sealed class ThresholdEstimate
    {
        public double WeakThresh;
        public double StrongThresh;
        /// <summary>true = 这两个阈值是可信建议值; false = 图本身给不出可信建议</summary>
        public bool Ok;
        /// <summary>参与训练的有效像素数</summary>
        public int MaskPixels;
        public int RequestedFeatures;
        public int Candidates;              // 该阈值下的候选点数
        public int Features;                // 该阈值下实际取到的特征点数
        public double MedianGradient;
        public double P95Gradient;
        /// <summary>自匹配自检得分, &lt; 0 表示未计算</summary>
        public double SelfScore;
        /// <summary>提示语 (含诊断数据与兜底建议)</summary>
        public string Note;
        /// <summary>特征点标注图宽 (等于 ROI 宽)</summary>
        public int FeaturesWidth;
        public int FeaturesHeight;
        public int FeaturesChannels;        // 与输入一致: 1 灰度 / 3 BGR
        /// <summary>特征点标注图像素 (行优先, 已去掉行对齐)。
        /// 在 EstimateThresholds 返回前就拷进托管内存了, 句柄释放后依然可用。</summary>
        public byte[] FeaturesImage;
    }

    /// <summary>匹配结果</summary>
    public sealed class MatchResult
    {
        public string ClassId;
        public int TemplateId;
        public double Score;
        public double X, Y;
        public bool IcpRefined;
        public double[] Box;                // 旋转外框 4 个角点: x,y,x,y,...
        public double[] Features;           // 匹配到的特征点: x,y,...
        public double RefinedX, RefinedY, RefinedAngle;
        public double Fitness;
        public double Overlap;
        public double[] GraspPoints;        // 抓取点: x,y,... (第 1 个即主抓取点)
    }

    /// <summary>matcher 句柄封装。非线程安全, 一个实例只在一个线程里用。</summary>
    public sealed class Matcher : IDisposable
    {
        private IntPtr _handle;

        public Matcher()
        {
            _handle = Native.sbm_create();
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("sbm_create 失败");
        }

        public void Clear()
        {
            Native.sbm_clear(_handle);
        }

        public int LoadedClassCount
        {
            get { return Native.sbm_loaded_class_count(_handle); }
        }

        public List<string> LoadedClassIds()
        {
            List<string> list = new List<string>();
            int n = LoadedClassCount;
            for (int i = 0; i < n; i++)
                list.Add(Utf8.PtrToString(Native.sbm_loaded_class_id(_handle, i)));
            return list;
        }

        /// <summary>自动探测某张训练图 (ROI 区域) 合适的弱/强阈值, 省去逐图手调。
        /// 建议在 Train 之前调用: 把返回的 WeakThresh / StrongThresh 填进 TrainParams 即可。
        /// Ok = false 时只给诊断说明, 不给建议值(此时 Weak/Strong 是默认的 30/60)。</summary>
        public ThresholdEstimate EstimateThresholds(RawImage image, int[] roi,
                                                    ThresholdSearchParams p, ExclusionZone[] zones,
                                                    RawImage? positiveMask, RawImage? negativeMask)
        {
            if (roi == null || roi.Length != 4) throw new ArgumentException("roi 必须是 [x, y, w, h]");
            // 调用方可能只改了少量字段就直接传进来, 金字塔数组得先备好
            if (p.PyramidLevels == null)
            {
                ThresholdSearchParams d = ThresholdSearchParams.Default();
                p.PyramidLevels = d.PyramidLevels;
                p.PyramidLevelCount = d.PyramidLevelCount;
            }

            IntPtr zonePtr = IntPtr.Zero;
            List<Utf8> zoneNames = new List<Utf8>();
            Native.SbmThresholdEstimate native = new Native.SbmThresholdEstimate();
            int rc;
            try
            {
                if (zones != null && zones.Length > 0)
                {
                    int size = Marshal.SizeOf(typeof(Native.SbmExclusionZone));
                    zonePtr = Marshal.AllocHGlobal(size * zones.Length);
                    for (int i = 0; i < zones.Length; i++)
                    {
                        Utf8 type = new Utf8(zones[i].Type ?? "exclude_rect");
                        zoneNames.Add(type);
                        Native.SbmExclusionZone z = new Native.SbmExclusionZone();
                        z.Type = type;
                        z.X = zones[i].X; z.Y = zones[i].Y; z.W = zones[i].W; z.H = zones[i].H;
                        Marshal.StructureToPtr(z, zonePtr + i * size, false);
                    }
                }

                using (UnmanagedStruct<RawImage> img = new UnmanagedStruct<RawImage>(image))
                using (UnmanagedStruct<RawImage> pos = UnmanagedStruct<RawImage>.Create(positiveMask))
                using (UnmanagedStruct<RawImage> neg = UnmanagedStruct<RawImage>.Create(negativeMask))
                using (UnmanagedStruct<Native.SbmThresholdEstimate> res =
                       new UnmanagedStruct<Native.SbmThresholdEstimate>())
                {
                    rc = Native.sbm_estimate_thresholds(_handle, img.Ptr, roi, ref p, zonePtr,
                                                        zones == null ? 0 : zones.Length,
                                                        pos.Ptr, neg.Ptr, res.Ptr);
                    native = res.Read();
                }
            }
            finally
            {
                foreach (Utf8 t in zoneNames) t.Dispose();
                if (zonePtr != IntPtr.Zero) Marshal.FreeHGlobal(zonePtr);
            }
            CheckResult(rc, "sbm_estimate_thresholds");

            // 标注图必须在这里立刻拷走: 它指向库内部缓冲, 句柄销毁后就失效了
            byte[] pixels = CopyImageBytes(native.FeaturesImage);
            int fw = native.FeaturesImage.Width;
            int fh = native.FeaturesImage.Height;
            int fch = native.FeaturesImage.Channels;
            if (fw <= 0 || fh <= 0) { fw = 0; fh = 0; fch = 0; }

            return new ThresholdEstimate
            {
                WeakThresh = native.WeakThresh,
                StrongThresh = native.StrongThresh,
                Ok = native.Ok != 0,
                MaskPixels = native.MaskPixels,
                RequestedFeatures = native.RequestedFeatures,
                Candidates = native.Candidates,
                Features = native.Features,
                MedianGradient = native.MedianGradient,
                P95Gradient = native.P95Gradient,
                SelfScore = native.SelfScore,
                Note = Utf8.PtrToString(native.Note),
                FeaturesWidth = fw,
                FeaturesHeight = fh,
                FeaturesChannels = fch,
                FeaturesImage = pixels,
            };
        }

        /// <summary>训练并保存模板。classId / saveDir 支持中文。</summary>
        public TrainResult Train(RawImage image, int[] roi, string classId, TrainParams p,
                                 string saveDir, ExclusionZone[] zones,
                                 RawImage? positiveMask, RawImage? negativeMask)
        {
            if (roi == null || roi.Length != 4) throw new ArgumentException("roi 必须是 [x, y, w, h]");

            IntPtr zonePtr = IntPtr.Zero;
            List<Utf8> zoneNames = new List<Utf8>();
            Native.SbmTrainResult native = new Native.SbmTrainResult();
            int rc;
            try
            {
                if (zones != null && zones.Length > 0)
                {
                    int size = Marshal.SizeOf(typeof(Native.SbmExclusionZone));
                    zonePtr = Marshal.AllocHGlobal(size * zones.Length);
                    for (int i = 0; i < zones.Length; i++)
                    {
                        Utf8 type = new Utf8(zones[i].Type ?? "exclude_rect");
                        zoneNames.Add(type);
                        Native.SbmExclusionZone z = new Native.SbmExclusionZone();
                        z.Type = type;
                        z.X = zones[i].X; z.Y = zones[i].Y; z.W = zones[i].W; z.H = zones[i].H;
                        Marshal.StructureToPtr(z, zonePtr + i * size, false);
                    }
                }

                using (Utf8 cid = new Utf8(classId))
                using (Utf8 dir = new Utf8(saveDir))
                using (UnmanagedStruct<RawImage> img = new UnmanagedStruct<RawImage>(image))
                using (UnmanagedStruct<RawImage> pos = UnmanagedStruct<RawImage>.Create(positiveMask))
                using (UnmanagedStruct<RawImage> neg = UnmanagedStruct<RawImage>.Create(negativeMask))
                using (UnmanagedStruct<Native.SbmTrainResult> res = new UnmanagedStruct<Native.SbmTrainResult>())
                {
                    rc = Native.sbm_train(_handle, img.Ptr, roi, cid, ref p, dir,
                                          zonePtr, zones == null ? 0 : zones.Length,
                                          pos.Ptr, neg.Ptr, res.Ptr);
                    native = res.Read();
                }
            }
            finally
            {
                foreach (Utf8 t in zoneNames) t.Dispose();
                if (zonePtr != IntPtr.Zero) Marshal.FreeHGlobal(zonePtr);
            }
            CheckResult(rc, "sbm_train");

            // 标注图必须在这里立刻拷走: 它指向库内部缓冲, 句柄销毁后就失效了
            byte[] pixels = CopyImageBytes(native.FeaturesImage);
            int fw = native.FeaturesImage.Width;
            int fh = native.FeaturesImage.Height;
            int fch = native.FeaturesImage.Channels;
            if (fw <= 0 || fh <= 0) { fw = 0; fh = 0; fch = 0; }

            return new TrainResult
            {
                YamlPath = Utf8.PtrToString(native.YamlPath),
                InfoPath = Utf8.PtrToString(native.InfoPath),
                PreviewPath = Utf8.PtrToString(native.PreviewPath),
                SaveDir = Utf8.PtrToString(native.SaveDir),
                BaseName = Utf8.PtrToString(native.BaseName),
                TemplateCount = native.TemplateCount,
                FeaturesWidth = fw,
                FeaturesHeight = fh,
                FeaturesChannels = fch,
                FeaturesImage = pixels,
            };
        }

        /// <summary>加载模板类别; path 可为 xxx.yaml / xxx.info.json / xxx.json。返回 class_id。</summary>
        public string AddTemplateClass(string path, TrainParams? overrideParams,
                                      out TrainParams finalParams)
        {
            TrainParams final = new TrainParams();
            IntPtr idPtr;
            using (Utf8 p = new Utf8(path))
            {
                if (overrideParams.HasValue)
                {
                    TrainParams ov = overrideParams.Value;
                    idPtr = Native.sbm_add_template_class_override(_handle, p, ref ov, out final);
                }
                else
                {
                    idPtr = Native.sbm_add_template_class_none(_handle, p, IntPtr.Zero, out final);
                }
            }
            string classId = Utf8.PtrToString(idPtr);
            if (classId == null) throw new InvalidOperationException("加载模板失败: " + LastError);
            finalParams = final;
            return classId;
        }

        /// <summary>设置某类别的抓取点 (ROI 坐标), xy = x,y,x,y,...; 传 null 清除。</summary>
        public void SetGraspPoints(string classId, double[] xy)
        {
            using (Utf8 c = new Utf8(classId))
            {
                if (xy == null || xy.Length < 2)
                {
                    CheckResult(Native.sbm_set_grasp_points(_handle, c, IntPtr.Zero, 0),
                                "sbm_set_grasp_points");
                    return;
                }
                GCHandle h = GCHandle.Alloc(xy, GCHandleType.Pinned);
                try
                {
                    CheckResult(Native.sbm_set_grasp_points(_handle, c, h.AddrOfPinnedObject(),
                                                            xy.Length / 2), "sbm_set_grasp_points");
                }
                finally { h.Free(); }
            }
        }

        /// <summary>执行匹配。classIds 传 null 表示匹配全部已加载类别。</summary>
        public List<MatchResult> Match(RawImage image, double scoreThreshold, string[] classIds,
                                       bool useNms, double nmsThreshold,
                                       int maxMatches, double minFitness, bool useRefine,
                                       double maxOverlap, RawImage? masks, bool fillOverlap)
        {
            IntPtr idsPtr = IntPtr.Zero;
            List<Utf8> idBufs = new List<Utf8>();
            int n;
            try
            {
                int idCount = 0;
                if (classIds != null && classIds.Length > 0)
                {
                    IntPtr[] ptrs = new IntPtr[classIds.Length];
                    for (int i = 0; i < classIds.Length; i++)
                    {
                        Utf8 u = new Utf8(classIds[i]);
                        idBufs.Add(u);
                        ptrs[i] = u;
                    }
                    idsPtr = Marshal.AllocHGlobal(IntPtr.Size * ptrs.Length);
                    Marshal.Copy(ptrs, 0, idsPtr, ptrs.Length);
                    idCount = ptrs.Length;
                }

                using (UnmanagedStruct<RawImage> img = new UnmanagedStruct<RawImage>(image))
                using (UnmanagedStruct<RawImage> msk = UnmanagedStruct<RawImage>.Create(masks))
                {
                    n = Native.sbm_match(_handle, img.Ptr, scoreThreshold, idsPtr, idCount,
                                         useNms ? 1 : 0, nmsThreshold, maxMatches, minFitness,
                                         useRefine ? 1 : 0, maxOverlap, msk.Ptr, fillOverlap ? 1 : 0);
                }
                CheckResult(n, "sbm_match");

                List<MatchResult> list = new List<MatchResult>(n);
                for (int i = 0; i < n; i++)
                {
                    Native.SbmMatchResult raw = new Native.SbmMatchResult();
                    CheckResult(Native.sbm_match_result(_handle, i, out raw), "sbm_match_result");
                    MatchResult m = new MatchResult();
                    m.ClassId = Utf8.PtrToString(raw.ClassId);
                    m.TemplateId = raw.TemplateId;
                    m.Score = raw.Score;
                    m.X = raw.X;
                    m.Y = raw.Y;
                    m.IcpRefined = raw.IcpRefined != 0;
                    m.Box = new double[8];
                    for (int k = 0; k < 8; k++) m.Box[k] = raw.Box[k];
                    m.Features = CopyDoubles(raw.Features, raw.FeatureCount * 2);
                    m.RefinedX = raw.RefinedX;
                    m.RefinedY = raw.RefinedY;
                    m.RefinedAngle = raw.RefinedAngle;
                    m.Fitness = raw.Fitness;
                    m.Overlap = raw.Overlap;
                    m.GraspPoints = CopyDoubles(raw.GraspPoints, raw.GraspCount * 2);
                    list.Add(m);
                }
                return list;
            }
            finally
            {
                foreach (Utf8 u in idBufs) u.Dispose();
                if (idsPtr != IntPtr.Zero) Marshal.FreeHGlobal(idsPtr);
            }
        }

        /// <summary>最近一次错误 (UTF-8, 无错误时为 null)</summary>
        public string LastError
        {
            get { return Utf8.PtrToString(Native.sbm_last_error(_handle)); }
        }

        private static double[] CopyDoubles(IntPtr ptr, int count)
        {
            if (ptr == IntPtr.Zero || count <= 0) return new double[0];
            double[] buf = new double[count];
            Marshal.Copy(ptr, buf, 0, count);
            return buf;
        }

        /// <summary>把库内部的图像缓冲拷进托管字节数组 (顺手去掉行对齐填充)</summary>
        private static byte[] CopyImageBytes(RawImage img)
        {
            if (img.Data == IntPtr.Zero || img.Width <= 0 || img.Height <= 0) return null;
            int ch = img.Channels <= 0 ? 1 : img.Channels;
            int rowBytes = img.Width * ch;
            int step = img.Step > 0 ? img.Step : rowBytes;
            byte[] raw = new byte[step * img.Height];
            Marshal.Copy(img.Data, raw, 0, raw.Length);
            if (step == rowBytes) return raw;
            byte[] pixels = new byte[rowBytes * img.Height];
            for (int r = 0; r < img.Height; r++)
                Buffer.BlockCopy(raw, r * step, pixels, r * rowBytes, rowBytes);
            return pixels;
        }

        private void CheckResult(int code, string api)
        {
            if (code < 0)
                throw new InvalidOperationException(
                    string.Format("{0} 失败 (code={1}): {2}", api, code, LastError));
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                Native.sbm_destroy(_handle);
                _handle = IntPtr.Zero;
            }
        }

#if SBM_OPENCVSHARP
        /// <summary>OpenCvSharp.Mat -> RawImage (要求连续内存)</summary>
        public static RawImage FromMat(OpenCvSharp.Mat mat)
        {
            if (mat == null || mat.Empty()) throw new ArgumentException("Mat 为空");
            if (!mat.IsContinuous()) throw new ArgumentException("Mat 必须连续, 先 Clone()");
            return new RawImage
            {
                Data = mat.Data,
                Width = mat.Width,
                Height = mat.Height,
                Channels = mat.Channels(),
                Step = (int)mat.Step(),
            };
        }
#endif
    }

    // ============================ P/Invoke 声明 ============================

    internal static class Native
    {
        internal const string Dll = "shape_based_matching_c";
        internal const CallingConvention Call = CallingConvention.Cdecl;

        [StructLayout(LayoutKind.Sequential)]
        internal struct SbmExclusionZone
        {
            public IntPtr Type;
            public int X, Y, W, H;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct SbmTrainResult
        {
            public IntPtr YamlPath, InfoPath, PreviewPath, SaveDir, BaseName;
            public int TemplateCount;
            public RawImage FeaturesImage;
        }

        [StructLayout(LayoutKind.Sequential)]
        internal struct SbmMatchResult
        {
            public IntPtr ClassId;
            public int TemplateId;
            public double Score, X, Y;
            public int IcpRefined;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 8)]
            public double[] Box;
            public int FeatureCount;
            public IntPtr Features;
            public double RefinedX, RefinedY, RefinedAngle, Fitness, Overlap;
            public int GraspCount;
            public IntPtr GraspPoints;
        }

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern IntPtr sbm_create();

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern void sbm_destroy(IntPtr h);

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern void sbm_clear(IntPtr h);

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern void sbm_train_params_init(ref TrainParams p);

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern int sbm_train(IntPtr h, IntPtr image, int[] roi, IntPtr classId,
                                             ref TrainParams p, IntPtr saveDir,
                                             IntPtr zones, int zoneCount,
                                             IntPtr positiveMask, IntPtr negativeMask,
                                             IntPtr outResult);

        // 同一个 C 函数的两种签名: 带覆盖参数 / 不覆盖(传 NULL)
        [DllImport(Dll, EntryPoint = "sbm_add_template_class", CallingConvention = Call)]
        internal static extern IntPtr sbm_add_template_class_override(IntPtr h, IntPtr path,
                                                                      ref TrainParams overrideParams,
                                                                      out TrainParams finalParams);

        [DllImport(Dll, EntryPoint = "sbm_add_template_class", CallingConvention = Call)]
        internal static extern IntPtr sbm_add_template_class_none(IntPtr h, IntPtr path,
                                                                  IntPtr overrideNull,
                                                                  out TrainParams finalParams);

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern int sbm_loaded_class_count(IntPtr h);

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern IntPtr sbm_loaded_class_id(IntPtr h, int index);

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern int sbm_set_grasp_points(IntPtr h, IntPtr classId, IntPtr xy, int count);

        [StructLayout(LayoutKind.Sequential)]
        internal struct SbmThresholdEstimate
        {
            public double WeakThresh, StrongThresh;
            [MarshalAs(UnmanagedType.I4)]
            public int Ok;
            public int MaskPixels, RequestedFeatures, Candidates, Features;
            public double MedianGradient, P95Gradient, SelfScore;
            public IntPtr Note;
            public RawImage FeaturesImage;
        }

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern void sbm_threshold_search_params_init(ref ThresholdSearchParams p);

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern int sbm_estimate_thresholds(IntPtr h, IntPtr image, int[] roi,
                                                           ref ThresholdSearchParams p,
                                                           IntPtr zones, int zoneCount,
                                                           IntPtr positiveMask, IntPtr negativeMask,
                                                           IntPtr outEstimate);

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern int sbm_match(IntPtr h, IntPtr image, double scoreThreshold,
                                             IntPtr classIds, int classIdCount,
                                             int useNms, double nmsThreshold,
                                             int maxMatches, double minFitness, int useRefine,
                                             double maxOverlap, IntPtr masks, int fillOverlap);

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern int sbm_match_result(IntPtr h, int index, out SbmMatchResult outResult);

        [DllImport(Dll, CallingConvention = Call)]
        internal static extern IntPtr sbm_last_error(IntPtr h);
    }

    // ============================ 非托管内存小工具 ============================

    /// <summary>把结构体固定到非托管内存, using 结束自动释放</summary>
    internal sealed class UnmanagedStruct<T> : IDisposable where T : struct
    {
        private IntPtr _ptr;
        private bool _disposed;

        public UnmanagedStruct() : this(default(T)) { }

        public UnmanagedStruct(T value)
        {
            _ptr = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(T)));
            Marshal.StructureToPtr(value, _ptr, false);
        }

        /// <summary>Nullable 版本: 没有值时用默认值填一份 (尺寸仍合法)</summary>
        public static UnmanagedStruct<T> Create(T? value)
        {
            return new UnmanagedStruct<T>(value.HasValue ? value.Value : default(T));
        }

        public IntPtr Ptr
        {
            get { return _ptr; }
        }

        public T Read()
        {
            return (T)Marshal.PtrToStructure(_ptr, typeof(T));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_ptr != IntPtr.Zero)
            {
                Marshal.DestroyStructure(_ptr, typeof(T));
                Marshal.FreeHGlobal(_ptr);
                _ptr = IntPtr.Zero;
            }
        }
    }

    /// <summary>把 string 按 UTF-8 拷到非托管内存, using 结束自动释放</summary>
    internal sealed class Utf8 : IDisposable
    {
        private IntPtr _ptr = IntPtr.Zero;
        private bool _disposed;

        public Utf8(string s)
        {
            if (s == null) return;
            byte[] bytes = Encoding.UTF8.GetBytes(s);
            _ptr = Marshal.AllocHGlobal(bytes.Length + 1);
            Marshal.Copy(bytes, 0, _ptr, bytes.Length);
            Marshal.WriteByte(_ptr, bytes.Length, 0);
        }

        public static implicit operator IntPtr(Utf8 u)
        {
            return u == null ? IntPtr.Zero : u._ptr;
        }

        /// <summary>库返回的 const char* (UTF-8) -> string</summary>
        public static string PtrToString(IntPtr p)
        {
            if (p == IntPtr.Zero) return null;
            int len = 0;
            while (Marshal.ReadByte(p, len) != 0) len++;
            if (len == 0) return string.Empty;
            byte[] buf = new byte[len];
            Marshal.Copy(p, buf, 0, len);
            return Encoding.UTF8.GetString(buf);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_ptr != IntPtr.Zero)
            {
                Marshal.FreeHGlobal(_ptr);
                _ptr = IntPtr.Zero;
            }
        }
    }
}
