import {
  Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";

type IntervalDatum = {
  date: string;
  point: number;
  lower: number;
  band: number;
  upper: number;
};

type ConformalChartProps = {
  data: IntervalDatum[];
  domain: [number, number];
  unit: string;
  label: string;
};

function ConformalChart({ data, domain, unit, label }: ConformalChartProps) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 16, right: 12, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id="intervalFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#16a34a" stopOpacity={0.34} />
            <stop offset="100%" stopColor="#38cf72" stopOpacity={0.06} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke="#eef0ec" vertical={false} />
        <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fill: "#92998f", fontSize: 11 }} dy={9} />
        <YAxis axisLine={false} tickLine={false} tick={{ fill: "#92998f", fontSize: 11 }} tickFormatter={(value) => `${Number(value).toFixed(Number.isInteger(Number(value)) || Number(value) >= 100 ? 0 : 1)} ${unit}`} domain={domain} allowDataOverflow />
        <Tooltip contentStyle={{ border: "1px solid #e8ebe5", borderRadius: 10, fontSize: 12, boxShadow: "0 8px 24px #29342112" }} />
        <Area dataKey="lower" stackId="interval" stroke="none" fill="transparent" name="Lower bound" />
        <Area dataKey="band" stackId="interval" stroke="none" fill="url(#intervalFill)" name={`${label} interval`} />
        <Area dataKey="point" stroke="#16a34a" strokeWidth={3} fill="none" name="Point estimate" dot={{ r: 4, fill: "#16a34a", stroke: "white", strokeWidth: 2 }} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export default ConformalChart;
