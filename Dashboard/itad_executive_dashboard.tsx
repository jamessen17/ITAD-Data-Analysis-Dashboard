import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { TrendingUp, DollarSign, Recycle, Leaf, Package, Clock, Award, Users } from 'lucide-react';

const ExecutiveDashboard = () => {
  // Sample data - in real implementation, this would come from CSV files
  const [dashboardData, setDashboardData] = useState({
    kpis: {
      totalRevenue: 2456780,
      carbonSaved: 145234,
      assetsProcessed: 4251,
      successRate: 82.3,
      avgProcessingTime: 5.2,
      customerSatisfaction: 4.6,
      circularValue: 1847562,
      materialRecovered: 28543
    },
    revenueByMonth: [
      { month: 'Jan', revenue: 187000, volume: 285 },
      { month: 'Feb', revenue: 195000, volume: 310 },
      { month: 'Mar', revenue: 218000, volume: 342 },
      { month: 'Apr', revenue: 201000, volume: 318 },
      { month: 'May', revenue: 234000, volume: 367 },
      { month: 'Jun', revenue: 189000, volume: 301 },
      { month: 'Jul', revenue: 176000, volume: 278 },
      { month: 'Aug', revenue: 198000, volume: 315 },
      { month: 'Sep', revenue: 267000, volume: 421 },
      { month: 'Oct', revenue: 298000, volume: 468 },
      { month: 'Nov', revenue: 341000, volume: 536 },
      { month: 'Dec', revenue: 352780, volume: 570 }
    ],
    devicePerformance: [
      { device: 'Laptop', processed: 1702, successful: 1395, successRate: 82, avgPrice: 385, carbonSaved: 54218 },
      { device: 'Desktop', processed: 1063, successful: 829, successRate: 78, avgPrice: 295, carbonSaved: 28654 },
      { device: 'Server', processed: 638, successful: 415, successRate: 65, avgPrice: 1250, carbonSaved: 38421 },
      { device: 'Monitor', processed: 425, successful: 361, successRate: 85, avgPrice: 145, carbonSaved: 12847 },
      { device: 'Tablet', processed: 340, successful: 255, successRate: 75, avgPrice: 180, carbonSaved: 8634 },
      { device: 'Smartphone', processed: 83, successful: 58, successRate: 70, avgPrice: 125, carbonSaved: 2460 }
    ],
    customerSegments: [
      { segment: 'Enterprise', value: 1105051, percentage: 45 },
      { segment: 'SMB', value: 614195, percentage: 25 },
      { segment: 'Education', value: 368517, percentage: 15 },
      { segment: 'Government', value: 245678, percentage: 10 },
      { segment: 'Consumer', value: 122839, percentage: 5 }
    ],
    environmentalTrend: [
      { quarter: 'Q1 2023', carbonSaved: 31245, materialsRecovered: 6234 },
      { quarter: 'Q2 2023', carbonSaved: 33567, materialsRecovered: 6789 },
      { quarter: 'Q3 2023', carbonSaved: 29834, materialsRecovered: 6156 },
      { quarter: 'Q4 2023', carbonSaved: 38912, materialsRecovered: 7845 },
      { quarter: 'Q1 2024', carbonSaved: 41276, materialsRecovered: 8234 },
      { quarter: 'Q2 2024', carbonSaved: 45123, materialsRecovered: 8976 },
      { quarter: 'Q3 2024', carbonSaved: 42567, materialsRecovered: 8542 },
      { quarter: 'Q4 2024', carbonSaved: 48734, materialsRecovered: 9567 }
    ],
    operationalMetrics: [
      { metric: 'Avg Processing Time', value: 5.2, unit: 'days', trend: -8.3 },
      { metric: 'Labor Cost per Unit', value: 87, unit: '$', trend: -12.1 },
      { metric: 'Parts Cost Ratio', value: 15.3, unit: '%', trend: -5.7 },
      { metric: 'Quality Score', value: 8.4, unit: '/10', trend: 11.2 }
    ],
    certificationBreakdown: [
      { certification: 'R2', count: 1698, percentage: 40 },
      { certification: 'e-Stewards', count: 1274, percentage: 30 },
      { certification: 'ISO14001', count: 850, percentage: 20 },
      { certification: 'None', value: 425, percentage: 10 }
    ]
  });

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4'];

  const KPICard = ({ title, value, unit, icon: Icon, trend, subtitle }) => (
    <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <div className="p-3 rounded-lg bg-blue-50">
          <Icon className="h-6 w-6 text-blue-600" />
        </div>
        {trend && (
          <div className={`flex items-center text-sm ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
            <TrendingUp className={`h-4 w-4 mr-1 ${trend < 0 ? 'rotate-180' : ''}`} />
            {Math.abs(trend)}%
          </div>
        )}
      </div>
      <div>
        <p className="text-2xl font-bold text-gray-900">{value.toLocaleString()}{unit}</p>
        <p className="text-sm text-gray-500 mt-1">{title}</p>
        {subtitle && <p className="text-xs text-gray-400 mt-1">{subtitle}</p>}
      </div>
    </div>
  );

  const MetricCard = ({ title, children }) => (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      {children}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">ITAD Executive Dashboard</h1>
        <p className="text-gray-600">Comprehensive view of IT Asset Disposition performance and environmental impact</p>
        <div className="text-sm text-gray-500 mt-2">Last updated: {new Date().toLocaleString()}</div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <KPICard
          title="Total Revenue"
          value={dashboardData.kpis.totalRevenue}
          unit=""
          icon={DollarSign}
          trend={15.7}
          subtitle="Year to date"
        />
        <KPICard
          title="Carbon Saved"
          value={dashboardData.kpis.carbonSaved}
          unit=" kg CO₂e"
          icon={Leaf}
          trend={23.4}
          subtitle="Equivalent to 315 cars off road"
        />
        <KPICard
          title="Assets Processed"
          value={dashboardData.kpis.assetsProcessed}
          unit=""
          icon={Package}
          trend={12.1}
          subtitle="85% of total intake"
        />
        <KPICard
          title="Success Rate"
          value={dashboardData.kpis.successRate}
          unit="%"
          icon={Award}
          trend={3.2}
          subtitle="Industry benchmark: 78%"
        />
      </div>

      {/* Revenue and Volume Trends */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <MetricCard title="Revenue & Volume Trends">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={dashboardData.revenueByMonth}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="month" tick={{fontSize: 12}} />
              <YAxis yAxisId="revenue" orientation="left" tick={{fontSize: 12}} />
              <YAxis yAxisId="volume" orientation="right" tick={{fontSize: 12}} />
              <Tooltip 
                formatter={(value, name) => [
                  name === 'revenue' ? `$${value.toLocaleString()}` : `${value} units`,
                  name === 'revenue' ? 'Revenue' : 'Volume'
                ]}
              />
              <Legend />
              <Bar yAxisId="volume" dataKey="volume" fill="#E5E7EB" name="Volume" opacity={0.7} />
              <Line 
                yAxisId="revenue" 
                type="monotone" 
                dataKey="revenue" 
                stroke="#3B82F6" 
                strokeWidth={3}
                name="Revenue"
                dot={{fill: '#3B82F6', strokeWidth: 2, r: 4}}
              />
            </LineChart>
          </ResponsiveContainer>
        </MetricCard>

        <MetricCard title="Customer Segment Revenue">
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={dashboardData.customerSegments}
                cx="50%"
                cy="50%"
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({segment, percentage}) => `${segment} (${percentage}%)`}
              >
                {dashboardData.customerSegments.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [`$${value.toLocaleString()}`, 'Revenue']} />
            </PieChart>
          </ResponsiveContainer>
        </MetricCard>
      </div>

      {/* Device Performance Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <MetricCard title="Device Performance by Type">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={dashboardData.devicePerformance}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="device" tick={{fontSize: 11}} />
              <YAxis yAxisId="count" orientation="left" tick={{fontSize: 12}} />
              <YAxis yAxisId="rate" orientation="right" tick={{fontSize: 12}} />
              <Tooltip />
              <Legend />
              <Bar yAxisId="count" dataKey="processed" fill="#93C5FD" name="Processed" />
              <Bar yAxisId="count" dataKey="successful" fill="#3B82F6" name="Successful" />
              <Line yAxisId="rate" type="monotone" dataKey="successRate" stroke="#10B981" strokeWidth={2} name="Success Rate %" />
            </BarChart>
          </ResponsiveContainer>
        </MetricCard>

        <MetricCard title="Environmental Impact Trends">
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={dashboardData.environmentalTrend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="quarter" tick={{fontSize: 11}} />
              <YAxis tick={{fontSize: 12}} />
              <Tooltip 
                formatter={(value, name) => [
                  name === 'carbonSaved' ? `${value.toLocaleString()} kg CO₂e` : `${value.toLocaleString()} kg`,
                  name === 'carbonSaved' ? 'Carbon Saved' : 'Materials Recovered'
                ]}
              />
              <Legend />
              <Area
                type="monotone"
                dataKey="carbonSaved"
                stackId="1"
                stroke="#10B981"
                fill="#10B981"
                fillOpacity={0.6}
                name="Carbon Saved"
              />
              <Area
                type="monotone"
                dataKey="materialsRecovered"
                stackId="2"
                stroke="#3B82F6"
                fill="#3B82F6"
                fillOpacity={0.4}
                name="Materials Recovered"
              />
            </AreaChart>
          </ResponsiveContainer>
        </MetricCard>
      </div>

      {/* Operational Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <MetricCard title="Operational KPIs">
          <div className="space-y-4">
            {dashboardData.operationalMetrics.map((metric, index) => (
              <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="text-sm font-medium text-gray-900">{metric.metric}</p>
                  <p className="text-lg font-semibold text-blue-600">{metric.value}{metric.unit}</p>
                </div>
                <div className={`flex items-center text-sm ${metric.trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
                  <TrendingUp className={`h-4 w-4 mr-1 ${metric.trend < 0 ? 'rotate-180' : ''}`} />
                  {Math.abs(metric.trend)}%
                </div>
              </div>
            ))}
          </div>
        </MetricCard>

        <MetricCard title="Key Performance Indicators">
          <div className="space-y-6">
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-gray-700">Processing Efficiency</span>
                <span className="text-sm text-gray-500">85%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{width: '85%'}}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-gray-700">Customer Satisfaction</span>
                <span className="text-sm text-gray-500">4.6/5.0</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-green-500 h-2 rounded-full" style={{width: '92%'}}></div>
              </div>
            </div>
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-gray-700">Asset Recovery Rate</span>
                <span className="text-sm text-gray-500">78%</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-yellow-500 h-2 rounded-full" style={{width: '78%'}}></div>
              </div>
            </div>
          </div>
        </MetricCard>

        <MetricCard title="Secondary KPIs">
          <div className="grid grid-cols-2 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <Clock className="h-8 w-8 text-blue-600 mx-auto mb-2" />
              <p className="text-2xl font-bold text-blue-600">{dashboardData.kpis.avgProcessingTime}</p>
              <p className="text-xs text-gray-600">Avg Processing Days</p>
            </div>
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <Users className="h-8 w-8 text-green-600 mx-auto mb-2" />
              <p className="text-2xl font-bold text-green-600">{dashboardData.kpis.customerSatisfaction}</p>
              <p className="text-xs text-gray-600">Customer Rating</p>
            </div>
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <Recycle className="h-8 w-8 text-purple-600 mx-auto mb-2" />
              <p className="text-2xl font-bold text-purple-600">{(dashboardData.kpis.materialRecovered/1000).toFixed(1)}t</p>
              <p className="text-xs text-gray-600">Materials Recovered</p>
            </div>
            <div className="text-center p-4 bg-yellow-50 rounded-lg">
              <DollarSign className="h-8 w-8 text-yellow-600 mx-auto mb-2" />
              <p className="text-2xl font-bold text-yellow-600">${(dashboardData.kpis.circularValue/1000000).toFixed(1)}M</p>
              <p className="text-xs text-gray-600">Circular Value</p>
            </div>
          </div>
        </MetricCard>
      </div>

      {/* Bottom Summary */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Executive Summary</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-green-600 mb-2">23.4%</div>
            <div className="text-sm text-gray-600">YoY Growth in Environmental Impact</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600 mb-2">$2.46M</div>
            <div className="text-sm text-gray-600">Total Revenue Generated</div>
          </div>
          <div className="text-center">
            <div className="text-3xl font-bold text-purple-600 mb-2">145.2t</div>
            <div className="text-sm text-gray-600">CO₂ Emissions Avoided</div>
          </div>
        </div>
        <div className="mt-6 p-4 bg-blue-50 rounded-lg">
          <p className="text-sm text-blue-800">
            <strong>Key Insight:</strong> Our ITAD operations have achieved an 82.3% refurbishment success rate, 
            generating $2.46M in revenue while saving 145,234 kg of CO₂ equivalent emissions. 
            The enterprise segment continues to drive 45% of total revenue, with Q4 showing strongest performance.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ExecutiveDashboard;